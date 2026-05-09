from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.db.sqlite_store import SQLiteStore
from backend.schemas.response import (
    ResponseActionItem,
    ResponseActionRequest,
    ResponseActionStatus,
    ResponseActionType,
)


@dataclass
class ActionResult:
    success: bool
    message: str
    rollback_data: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class PermissionError(Exception):
    pass


class ResponseEngine:
    def __init__(self, database_path: str, quarantine_dir: str = "quarantine"):
        self.store = SQLiteStore(database_path)
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_admin(self) -> None:
        if platform.system() == "Windows":
            try:
                import ctypes

                if not ctypes.windll.shell32.IsUserAnAdmin():
                    raise PermissionError("Administrator privileges are required for response actions.")
            except Exception as exc:
                raise PermissionError(str(exc)) from exc
        else:
            if os.geteuid() != 0:
                raise PermissionError("Root privileges are required for response actions.")

    def propose_action(self, request: ResponseActionRequest) -> int:
        now = datetime.utcnow().isoformat()
        with self.store.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO response_actions
                  (alert_id, action_type, target, parameters_json, status,
                   requested_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.alert_id,
                    request.action_type,
                    request.target,
                    json.dumps(request.parameters or {}),
                    ResponseActionStatus.pending.value,
                    request.requested_by,
                    now,
                    now,
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to queue response action.")
            action_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO action_audit (
                  action_id, event_type, actor, details_json, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    "requested",
                    request.requested_by,
                    json.dumps({"reason": request.reason, "parameters": request.parameters}),
                    now,
                ),
            )
        return action_id

    def approve_action(self, action_id: int, approved_by: str, approved: bool, justification: str | None = None) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        action = self.store.get_response_action(action_id)
        if action is None:
            raise ValueError("Action not found")
        if action["status"] != ResponseActionStatus.pending.value:
            raise ValueError("Action is not pending approval")
        status = ResponseActionStatus.approved.value if approved else ResponseActionStatus.canceled.value
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE response_actions SET status=?, approved_by=?, updated_at=? WHERE id=?",
                (status, approved_by, now, action_id),
            )
            conn.execute(
                "INSERT INTO action_audit (action_id, event_type, actor, details_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    action_id,
                    "approved" if approved else "canceled",
                    approved_by,
                    json.dumps({"justification": justification}),
                    now,
                ),
            )
        if not approved:
            return {"action_id": action_id, "status": status}

        result = self.execute_action(action)
        return {"action_id": action_id, "status": ResponseActionStatus.executed.value if result.success else ResponseActionStatus.failed.value, "message": result.message}

    def execute_action(self, action: dict[str, Any]) -> ActionResult:
        self._ensure_admin()
        action_type = action["action_type"]
        target = action["target"]
        params = json.loads(action["parameters_json"] or "{}")
        handler = self._get_handler(action_type)
        result = handler(target, params)
        now = datetime.utcnow().isoformat()
        status = ResponseActionStatus.executed.value if result.success else ResponseActionStatus.failed.value
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE response_actions SET status=?, result_json=?, rollback_json=?, updated_at=?, executed_at=? WHERE id=?",
                (
                    status,
                    json.dumps(result.details),
                    json.dumps(result.rollback_data),
                    now,
                    now,
                    action["id"],
                ),
            )
            conn.execute(
                "INSERT INTO action_audit (action_id, event_type, actor, details_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    action["id"],
                    "executed",
                    action.get("approved_by") or "system",
                    json.dumps(result.details),
                    now,
                ),
            )
        return result

    def rollback_action(self, action_id: int, requested_by: str, reason: str | None = None) -> dict[str, Any]:
        action = self.store.get_response_action(action_id)
        if action is None:
            raise ValueError("Action not found")
        if action["status"] != ResponseActionStatus.executed.value:
            raise ValueError("Only executed actions can be rolled back")
        rollback_data = json.loads(action.get("rollback_json") or "{}")
        if not rollback_data:
            raise ValueError("No rollback data available for this action")
        result = self._perform_rollback(action["action_type"], rollback_data)
        now = datetime.utcnow().isoformat()
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE response_actions SET status=?, updated_at=? WHERE id=?",
                (ResponseActionStatus.rolled_back.value, now, action_id),
            )
            conn.execute(
                "INSERT INTO action_audit (action_id, event_type, actor, details_json, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    action_id,
                    "rolled_back",
                    requested_by,
                    json.dumps({"reason": reason, "rollback_result": result.details}),
                    now,
                ),
            )
        return {"action_id": action_id, "rolled_back": result.success, "message": result.message}

    def get_pending_actions(self, limit: int = 50) -> list[ResponseActionItem]:
        rows = self.store.get_pending_response_actions(limit)
        return [ResponseActionItem(**row) for row in rows]

    def _get_handler(self, action_type: str):
        mapping = {
            ResponseActionType.block_ip.value: self._block_ip,
            ResponseActionType.kill_process.value: self._kill_process,
            ResponseActionType.disable_outbound.value: self._disable_outbound,
            ResponseActionType.quarantine_file.value: self._quarantine_file,
            ResponseActionType.windows_defender_exclusion.value: self._windows_defender_exclusion,
            ResponseActionType.disable_adapter.value: self._disable_adapter,
            ResponseActionType.forensic_snapshot.value: self._forensic_snapshot,
            ResponseActionType.incident_report.value: self._incident_report,
        }
        handler = mapping.get(action_type)
        if handler is None:
            raise ValueError(f"Unsupported action type: {action_type}")
        return handler

    def _run_command(self, command: list[str], check: bool = False) -> tuple[bool, str]:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=check)
            output = completed.stdout.strip() or completed.stderr.strip()
            return completed.returncode == 0, output
        except Exception as exc:
            return False, str(exc)

    def _block_ip(self, target: str, params: dict[str, Any]) -> ActionResult:
        if platform.system() == "Windows":
            rule_name = f"Safeguard-Block-{target}"
            command = [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name={rule_name}",
                "dir=in",
                "action=block",
                f"remoteip={target}",
            ]
            success, output = self._run_command(command)
            return ActionResult(success, output, rollback_data={"action": "remove_firewall_rule", "rule_name": rule_name})
        else:
            if shutil.which("ufw"):
                command = ["ufw", "deny", "from", target, "to", "any"]
            else:
                command = ["iptables", "-A", "INPUT", "-s", target, "-j", "DROP"]
            success, output = self._run_command(command)
            return ActionResult(success, output, rollback_data={"action": "remove_block_ip", "target": target})

    def _kill_process(self, target: str, params: dict[str, Any]) -> ActionResult:
        pid = int(target)
        if platform.system() == "Windows":
            command = ["taskkill", "/PID", str(pid), "/F"]
            success, output = self._run_command(command)
            return ActionResult(success, output, rollback_data={"action": "no_rollback"})
        else:
            try:
                os.kill(pid, 15)
                return ActionResult(True, f"SIGTERM sent to {pid}", rollback_data={"action": "no_rollback"})
            except ProcessLookupError:
                return ActionResult(False, f"Process {pid} not found", {})
            except PermissionError as exc:
                return ActionResult(False, str(exc), {})

    def _disable_outbound(self, target: str, params: dict[str, Any]) -> ActionResult:
        if platform.system() == "Windows":
            rule_name = f"Safeguard-Outbound-{target}"
            command = [
                "netsh",
                "advfirewall",
                "firewall",
                "add",
                "rule",
                f"name={rule_name}",
                "dir=out",
                "action=block",
                f"remoteip={target}",
            ]
            success, output = self._run_command(command)
            return ActionResult(success, output, rollback_data={"action": "remove_firewall_rule", "rule_name": rule_name})
        else:
            if shutil.which("ufw"):
                command = ["ufw", "deny", "out", "to", target]
            else:
                command = ["iptables", "-A", "OUTPUT", "-d", target, "-j", "REJECT"]
            success, output = self._run_command(command)
            return ActionResult(success, output, rollback_data={"action": "remove_outbound_block", "target": target})

    def _quarantine_file(self, target: str, params: dict[str, Any]) -> ActionResult:
        source = Path(target)
        if not source.exists():
            return ActionResult(False, "File not found", {})
        dest = self.quarantine_dir / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{source.name}"
        try:
            shutil.move(str(source), str(dest))
            return ActionResult(True, f"Quarantined {target}", rollback_data={"action": "restore_file", "original": str(source), "quarantine": str(dest)})
        except Exception as exc:
            return ActionResult(False, str(exc), {})

    def _windows_defender_exclusion(self, target: str, params: dict[str, Any]) -> ActionResult:
        if platform.system() != "Windows":
            return ActionResult(False, "Windows Defender actions are only supported on Windows.", {})
        command = [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Add-MpPreference -ExclusionPath '{target}'"
        ]
        success, output = self._run_command(command)
        return ActionResult(success, output, rollback_data={"action": "remove_defender_exclusion", "path": target})

    def _disable_adapter(self, target: str, params: dict[str, Any]) -> ActionResult:
        if platform.system() == "Windows":
            command = ["netsh", "interface", "set", "interface", target, "disabled"]
        else:
            command = ["ip", "link", "set", target, "down"]
        success, output = self._run_command(command)
        return ActionResult(success, output, rollback_data={"action": "enable_adapter", "adapter": target})

    def _forensic_snapshot(self, target: str, params: dict[str, Any]) -> ActionResult:
        snapshot_dir = self.quarantine_dir / "forensic"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_file = snapshot_dir / f"snapshot-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "platform": platform.system(),
            "target": target,
            "process_list": self._collect_process_list(),
            "network_state": self._collect_network_state(),
        }
        snapshot_file.write_text(json.dumps(data, indent=2))
        return ActionResult(True, f"Snapshot stored at {snapshot_file}", rollback_data={"action": "no_rollback", "snapshot_path": str(snapshot_file)}, details={"snapshot_path": str(snapshot_file)})

    def _incident_report(self, target: str, params: dict[str, Any]) -> ActionResult:
        report_dir = self.quarantine_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"report-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "target": target,
            "parameters": params,
            "summary": params.get("summary", "No summary provided"),
        }
        report_file.write_text(json.dumps(report, indent=2))
        return ActionResult(True, f"Incident report exported to {report_file}", rollback_data={"action": "no_rollback", "report_path": str(report_file)}, details={"report_path": str(report_file)})

    def _perform_rollback(self, action_type: str, rollback_data: dict[str, Any]) -> ActionResult:
        action = rollback_data.get("action")
        if action == "remove_firewall_rule":
            rule_name = rollback_data["rule_name"]
            if platform.system() == "Windows":
                command = ["netsh", "advfirewall", "firewall", "delete", "rule", f"name={rule_name}"]
            else:
                return ActionResult(False, "Firewall rollback only supported on Windows.", {})
            success, output = self._run_command(command)
            return ActionResult(success, output)
        if action == "remove_block_ip" or action == "remove_outbound_block":
            target = rollback_data["target"]
            if shutil.which("ufw"):
                command = ["ufw", "delete", "deny", "from", target]
            else:
                command = ["iptables", "-D", "INPUT", "-s", target, "-j", "DROP"] if action == "remove_block_ip" else ["iptables", "-D", "OUTPUT", "-d", target, "-j", "REJECT"]
            success, output = self._run_command(command)
            return ActionResult(success, output)
        if action == "restore_file":
            original = Path(rollback_data["original"])
            quarantine = Path(rollback_data["quarantine"])
            try:
                shutil.move(str(quarantine), str(original))
                return ActionResult(True, f"Restored {original}")
            except Exception as exc:
                return ActionResult(False, str(exc))
        if action == "enable_adapter":
            adapter = rollback_data["adapter"]
            if platform.system() == "Windows":
                command = ["netsh", "interface", "set", "interface", adapter, "enabled"]
            else:
                command = ["ip", "link", "set", adapter, "up"]
            success, output = self._run_command(command)
            return ActionResult(success, output)
        return ActionResult(False, "Rollback action not supported", {})

    def _collect_process_list(self) -> list[dict[str, Any]]:
        processes = []
        if platform.system() == "Windows":
            command = ["tasklist", "/FO", "CSV"]
        else:
            command = ["ps", "-eo", "pid,comm,user,etime"]
        success, output = self._run_command(command)
        if not success:
            return [{"error": output}]
        for line in output.splitlines()[1:]:
            parts = [part.strip('"') for part in line.split(",")]
            if len(parts) >= 4:
                processes.append({"pid": parts[0], "command": parts[1], "user": parts[2], "elapsed": parts[3]})
        return processes

    def _collect_network_state(self) -> dict[str, Any]:
        if platform.system() == "Windows":
            command = ["ipconfig", "/all"]
        else:
            command = ["ss", "-tunap"] if shutil.which("ss") else ["netstat", "-tunap"]
        success, output = self._run_command(command)
        return {"success": success, "output": output}
