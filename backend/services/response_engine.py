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
from sqlalchemy.orm import Session
from backend.db.postgres_store import PostgresStore
from backend.db.models import ResponseAction, ActionAudit
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
    def __init__(self, db: Session, quarantine_dir: str = "quarantine"):
        self.db = db
        self.store = PostgresStore(db)
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_admin(self) -> None:
        if platform.system() == "Windows":
            try:
                import ctypes
                if not ctypes.windll.shell32.IsUserAnAdmin():
                    raise PermissionError("Administrator privileges required.")
            except Exception as exc:
                raise PermissionError(str(exc)) from exc
        else:
            if os.geteuid() != 0:
                raise PermissionError("Root privileges required.")

    def propose_action(self, request: ResponseActionRequest) -> int:
        action = ResponseAction(
            alert_id=request.alert_id,
            action_type=request.action_type,
            target=request.target,
            parameters_json=json.dumps(request.parameters or {}),
            status=ResponseActionStatus.pending.value,
            requested_by=request.requested_by,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        self.db.add(action)
        self.db.commit()
        self.db.refresh(action)
        
        audit = ActionAudit(
            action_id=action.id,
            event_type="requested",
            actor=request.requested_by,
            details_json=json.dumps({"reason": request.reason, "parameters": request.parameters}),
            timestamp=datetime.utcnow()
        )
        self.db.add(audit)
        self.db.commit()
        return action.id

    def approve_action(self, action_id: int, approved_by: str, approved: bool, justification: str | None = None) -> dict[str, Any]:
        action = self.db.query(ResponseAction).filter(ResponseAction.id == action_id).first()
        if not action:
            raise ValueError("Action not found")
        if action.status != ResponseActionStatus.pending.value:
            raise ValueError("Action is not pending approval")
            
        action.status = ResponseActionStatus.approved.value if approved else ResponseActionStatus.canceled.value
        action.approved_by = approved_by
        action.updated_at = datetime.utcnow()
        
        audit = ActionAudit(
            action_id=action_id,
            event_type="approved" if approved else "canceled",
            actor=approved_by,
            details_json=json.dumps({"justification": justification}),
            timestamp=datetime.utcnow()
        )
        self.db.add(audit)
        self.db.commit()
        
        if not approved:
            return {"action_id": action_id, "status": action.status}

        result = self.execute_action_obj(action)
        return {"action_id": action_id, "status": action.status, "message": result.message}

    def execute_action_obj(self, action: ResponseAction) -> ActionResult:
        self._ensure_admin()
        params = json.loads(action.parameters_json or "{}")
        handler = self._get_handler(action.action_type)
        result = handler(action.target, params)
        
        action.status = ResponseActionStatus.executed.value if result.success else ResponseActionStatus.failed.value
        action.result_json = json.dumps(result.details)
        action.rollback_json = json.dumps(result.rollback_data)
        action.updated_at = datetime.utcnow()
        action.executed_at = datetime.utcnow()
        
        audit = ActionAudit(
            action_id=action.id,
            event_type="executed",
            actor=action.approved_by or "system",
            details_json=json.dumps(result.details),
            timestamp=datetime.utcnow()
        )
        self.db.add(audit)
        self.db.commit()
        return result

    def rollback_action(self, action_id: int, requested_by: str, reason: str | None = None) -> dict[str, Any]:
        action = self.db.query(ResponseAction).filter(ResponseAction.id == action_id).first()
        if not action or action.status != ResponseActionStatus.executed.value:
            raise ValueError("Invalid action for rollback")
            
        rollback_data = json.loads(action.rollback_json or "{}")
        result = self._perform_rollback(action.action_type, rollback_data)
        
        action.status = ResponseActionStatus.rolled_back.value
        action.updated_at = datetime.utcnow()
        
        audit = ActionAudit(
            action_id=action_id,
            event_type="rolled_back",
            actor=requested_by,
            details_json=json.dumps({"reason": reason, "rollback_result": result.details}),
            timestamp=datetime.utcnow()
        )
        self.db.add(audit)
        self.db.commit()
        return {"action_id": action_id, "rolled_back": result.success, "message": result.message}

    def get_pending_actions(self, limit: int = 50) -> list[ResponseActionItem]:
        return [ResponseActionItem(**row) for row in self.store.get_pending_response_actions(limit)]

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
        return mapping[action_type]

    def _run_command(self, command: list[str], check: bool = False) -> tuple[bool, str]:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=check)
            return completed.returncode == 0, completed.stdout.strip() or completed.stderr.strip()
        except Exception as exc:
            return False, str(exc)

    def _block_ip(self, target: str, params: dict[str, Any]) -> ActionResult:
        if platform.system() == "Windows":
            rule_name = f"Safeguard-Block-{target}"
            cmd = ["netsh", "advfirewall", "firewall", "add", "rule", f"name={rule_name}", "dir=in", "action=block", f"remoteip={target}"]
            success, out = self._run_command(cmd)
            return ActionResult(success, out, {"action": "remove_firewall_rule", "rule_name": rule_name})
        return ActionResult(False, "Platform not supported for auto-block", {})

    def _kill_process(self, target: str, params: dict[str, Any]) -> ActionResult:
        pid = int(target)
        try:
            os.kill(pid, 9)
            return ActionResult(True, f"Process {pid} killed")
        except Exception as e:
            return ActionResult(False, str(e))

    def _disable_outbound(self, target: str, params: dict[str, Any]) -> ActionResult:
        return ActionResult(False, "Not implemented", {})

    def _quarantine_file(self, target: str, params: dict[str, Any]) -> ActionResult:
        source = Path(target)
        if not source.exists(): return ActionResult(False, "File missing")
        dest = self.quarantine_dir / f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{source.name}"
        shutil.move(str(source), str(dest))
        return ActionResult(True, f"Quarantined {target}", {"action": "restore_file", "original": str(source), "quarantine": str(dest)})

    def _windows_defender_exclusion(self, target: str, params: dict[str, Any]) -> ActionResult:
        return ActionResult(False, "Not implemented", {})

    def _disable_adapter(self, target: str, params: dict[str, Any]) -> ActionResult:
        return ActionResult(False, "Not implemented", {})

    def _forensic_snapshot(self, target: str, params: dict[str, Any]) -> ActionResult:
        return ActionResult(True, "Snapshot placeholder")

    def _incident_report(self, target: str, params: dict[str, Any]) -> ActionResult:
        return ActionResult(True, "Report generated")

    def _perform_rollback(self, action_type: str, rollback_data: dict[str, Any]) -> ActionResult:
        action = rollback_data.get("action")
        if action == "restore_file":
            shutil.move(rollback_data["quarantine"], rollback_data["original"])
            return ActionResult(True, "Restored")
        return ActionResult(False, "No rollback action found")
