import asyncio
import socket
import ssl
import time
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import whois
import dns.resolver


class ActiveScanner:
    """Safe reconnaissance scanner for local and external targets."""

    COMMON_PORTS = {
        21: ("FTP", "File Transfer Protocol (transmits data in plain text)"),
        22: ("SSH", "Secure Shell (remote login)"),
        23: ("Telnet", "Unencrypted remote login (highly insecure)"),
        25: ("SMTP", "Simple Mail Transfer Protocol"),
        53: ("DNS", "Domain Name System"),
        80: ("HTTP", "Unencrypted web traffic"),
        110: ("POP3", "Post Office Protocol"),
        135: ("RPC", "Windows RPC"),
        139: ("NetBIOS", "Windows NetBIOS"),
        143: ("IMAP", "Internet Message Access Protocol"),
        443: ("HTTPS", "Secure web traffic"),
        445: ("SMB", "Windows File Sharing"),
        3306: ("MySQL", "Database"),
        3389: ("RDP", "Remote Desktop Protocol"),
        5432: ("PostgreSQL", "Database"),
        8000: ("HTTP-ALT", "Alternative web port"),
        8080: ("HTTP-Proxy", "Web proxy or alternative web port"),
    }

    async def scan_target(self, target: str) -> Dict[str, Any]:
        """Perform a full safe reconnaissance scan on a target."""
        results: Dict[str, Any] = {
            "target": target,
            "timestamp": datetime.utcnow().isoformat(),
            "dns": {},
            "ports": [],
            "ssl": None,
            "http_headers": {},
            "whois": None,
            "latency_ms": None,
            "error": None,
        }

        try:
            # 1. DNS Resolution & Latency
            results["dns"], results["latency_ms"] = await self._resolve_and_ping(target)
            ip_address = results["dns"].get("ip_address")

            if not ip_address:
                results["error"] = "Could not resolve target IP address."
                return results

            # 2. Port Scanning
            results["ports"] = await self._scan_ports(ip_address)

            # 3. SSL Check (if port 443 is open)
            if any(p["port"] == 443 for p in results["ports"]):
                results["ssl"] = await self._check_ssl(ip_address)

            # 4. HTTP Headers (if port 80 or 443 is open)
            if any(p["port"] in (80, 443, 8000, 8080) for p in results["ports"]):
                port_to_check = 443 if any(p["port"] == 443 for p in results["ports"]) else 80
                results["http_headers"] = await self._get_http_headers(ip_address, port_to_check)

            # 5. WHOIS (only if it's a domain)
            if not self._is_ip(target):
                results["whois"] = await self._get_whois(target)

            # 6. Advanced Vulnerability Configuration Checks
            results["security_configs"] = self._evaluate_security(
                results["ports"], 
                results["http_headers"], 
                results["dns"]
            )

        except Exception as e:
            results["error"] = str(e)

        return results

    def _is_ip(self, target: str) -> bool:
        try:
            socket.inet_aton(target)
            return True
        except socket.error:
            return False

    async def _resolve_and_ping(self, target: str) -> tuple[Dict[str, Any], Optional[float]]:
        loop = asyncio.get_event_loop()
        dns_info = {}
        latency = None
        ip_address = None

        start_time = time.time()
        try:
            # Basic resolution
            if self._is_ip(target):
                ip_address = target
                try:
                    hostnames = await loop.run_in_executor(None, socket.gethostbyaddr, target)
                    dns_info["hostname"] = hostnames[0]
                except socket.herror:
                    dns_info["hostname"] = None
            else:
                ip_address = await loop.run_in_executor(None, socket.gethostbyname, target)
                dns_info["hostname"] = target

            dns_info["ip_address"] = ip_address
            latency = round((time.time() - start_time) * 1000, 2)

            # Additional DNS records if it's a domain
            if not self._is_ip(target):
                for record_type in ['MX', 'TXT', 'NS']:
                    try:
                        answers = await loop.run_in_executor(None, dns.resolver.resolve, target, record_type)
                        dns_info[f"{record_type}_records"] = [str(rdata) for rdata in answers]
                    except Exception:
                        pass
        except Exception:
            pass

        return dns_info, latency

    async def _scan_ports(self, ip: str) -> List[Dict[str, Any]]:
        open_ports = []
        loop = asyncio.get_event_loop()

        async def check_port(port: int):
            try:
                conn = asyncio.open_connection(ip, port)
                reader, writer = await asyncio.wait_for(conn, timeout=1.0)
                
                banner = None
                if port not in (80, 443, 8000, 8080):  # HTTP doesn't send banner first
                    try:
                        data = await asyncio.wait_for(reader.read(1024), timeout=1.5)
                        if data:
                            banner = data.decode('utf-8', errors='ignore').strip()
                            # Truncate long banners
                            if len(banner) > 200:
                                banner = banner[:197] + "..."
                    except Exception:
                        pass

                writer.close()
                await writer.wait_closed()
                
                service_name, description = self.COMMON_PORTS.get(port, ("Unknown", "Unidentified service"))
                open_ports.append({
                    "port": port,
                    "service": service_name,
                    "description": description,
                    "banner": banner,
                    "vulnerability_context": self._get_port_context(port)
                })
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                pass

        # Scan top common ports
        tasks = [check_port(port) for port in self.COMMON_PORTS.keys()]
        await asyncio.gather(*tasks)
        return sorted(open_ports, key=lambda x: x["port"])

    def _get_port_context(self, port: int) -> str:
        contexts = {
            21: "FTP sends credentials in plain text. Vulnerable to sniffing.",
            22: "SSH is secure, but vulnerable to brute-force if passwords are weak. Ensure key-based auth is used.",
            23: "Telnet is highly insecure and obsolete. All traffic is unencrypted.",
            80: "HTTP is unencrypted. Susceptible to Man-in-the-Middle (MitM) attacks.",
            445: "SMB is frequently targeted by ransomware (e.g., WannaCry). Ensure it is not exposed to the public internet.",
            3389: "RDP is a prime target for ransomware operators. Should be behind a VPN."
        }
        return contexts.get(port, "Ensure this service is intentionally exposed and properly secured.")

    async def _check_ssl(self, ip: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        def get_cert():
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                with socket.create_connection((ip, 443), timeout=3) as sock:
                    with ctx.wrap_socket(sock, server_hostname=ip) as ssock:
                        cert = ssock.getpeercert(binary_form=True)
                        import cryptography.x509
                        from cryptography.hazmat.backends import default_backend
                        x509 = cryptography.x509.load_der_x509_certificate(cert, default_backend())
                        return {
                            "issuer": x509.issuer.rfc4514_string(),
                            "subject": x509.subject.rfc4514_string(),
                            "expires": x509.not_valid_after.isoformat() if x509.not_valid_after else None,
                        }
            except Exception:
                return None
        
        return await loop.run_in_executor(None, get_cert)

    async def _get_http_headers(self, ip: str, port: int) -> Dict[str, str]:
        protocol = "https" if port == 443 else "http"
        url = f"{protocol}://{ip}:{port}/"
        try:
            async with httpx.AsyncClient(verify=False, timeout=3.0) as client:
                response = await client.head(url, follow_redirects=True)
                return dict(response.headers)
        except Exception:
            return {}

    async def _get_whois(self, domain: str) -> Optional[Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        def query_whois():
            try:
                w = whois.whois(domain)
                return {
                    "registrar": w.registrar,
                    "creation_date": str(w.creation_date[0]) if isinstance(w.creation_date, list) else str(w.creation_date),
                    "expiration_date": str(w.expiration_date[0]) if isinstance(w.expiration_date, list) else str(w.expiration_date),
                    "name_servers": w.name_servers
                }
            except Exception:
                return None
                
        return await loop.run_in_executor(None, query_whois)

    def _evaluate_security(self, ports: List[Dict[str, Any]], headers: Dict[str, str], dns_info: Dict[str, Any]) -> List[Dict[str, str]]:
        vulns = []
        
        # 1. Bruteforce Vulnerability Checks
        open_port_nums = [p["port"] for p in ports]
        if 22 in open_port_nums:
            vulns.append({
                "type": "Brute-Force Risk",
                "severity": "High",
                "description": "SSH (Port 22) is publicly exposed. This service is highly targeted by automated brute-force scripts. Ensure key-based authentication is enforced and password login is disabled."
            })
        if 3389 in open_port_nums:
            vulns.append({
                "type": "Brute-Force & Ransomware Risk",
                "severity": "Critical",
                "description": "RDP (Port 3389) is exposed. This is the #1 vector for ransomware deployment. It should be placed behind a VPN."
            })
        if 21 in open_port_nums or 23 in open_port_nums:
            vulns.append({
                "type": "Cleartext Authentication",
                "severity": "Critical",
                "description": "FTP (21) or Telnet (23) is exposed. Passwords are sent in cleartext and can be easily intercepted (sniffed) on the network."
            })

        # 2. DDoS Vulnerability Check
        if headers:
            headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
            is_protected = False
            waf_signatures = ['cloudflare', 'cloudfront', 'akamai', 'sucuri', 'incapsula', 'fastly']
            
            # Check Server header or custom headers
            server_header = headers_lower.get('server', '')
            if any(waf in server_header for waf in waf_signatures):
                is_protected = True
            if 'x-sucuri-id' in headers_lower or 'x-amz-cf-id' in headers_lower or 'cf-ray' in headers_lower:
                is_protected = True
                
            if not is_protected:
                vulns.append({
                    "type": "DDoS Vulnerability",
                    "severity": "High",
                    "description": "No Web Application Firewall (WAF) or CDN detected. The origin IP is directly exposed, making the server highly susceptible to volumetric Layer 3/4 and Layer 7 DDoS attacks."
                })

        # 3. Web Security Headers (XSS, Clickjacking, MitM)
        if headers:
            headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
            
            if 'strict-transport-security' not in headers_lower and 443 in open_port_nums:
                vulns.append({
                    "type": "Missing HSTS",
                    "severity": "Medium",
                    "description": "Strict-Transport-Security header is missing. The site is vulnerable to Man-in-the-Middle (MitM) attacks stripping SSL (SSL Stripping)."
                })
            
            if 'content-security-policy' not in headers_lower:
                vulns.append({
                    "type": "Missing CSP",
                    "severity": "Medium",
                    "description": "Content-Security-Policy header is missing. This significantly increases the risk of Cross-Site Scripting (XSS) attacks succeeding."
                })
                
            if 'x-frame-options' not in headers_lower:
                vulns.append({
                    "type": "Clickjacking Risk",
                    "severity": "Low",
                    "description": "X-Frame-Options header is missing. The site can be embedded in an iframe on a malicious website, enabling Clickjacking attacks."
                })

        # Return a clean list of findings. If empty, the target has a good basic posture.
        return vulns
