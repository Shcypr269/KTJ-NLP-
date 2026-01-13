"""
Action Executor Module - Handles enterprise action execution
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    FILE_TICKET = "file_ticket"
    SCHEDULE_MEETING = "schedule_meeting"
    REQUEST_SOFTWARE = "request_software"
    ESCALATE_ISSUE = "escalate_issue"
    APPLY_LEAVE = "apply_leave"
    UPDATE_DOCUMENTATION = "update_documentation"


class ActionExecutor:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.action_log = []
    
    def execute_action(self, action_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.now().isoformat()
        action_id = f"ACT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if action_type == ActionType.FILE_TICKET.value:
                result = self._file_ticket(parameters)
            elif action_type == ActionType.SCHEDULE_MEETING.value:
                result = self._schedule_meeting(parameters)
            elif action_type == ActionType.REQUEST_SOFTWARE.value:
                result = self._request_software(parameters)
            elif action_type == ActionType.ESCALATE_ISSUE.value:
                result = self._escalate_issue(parameters)
            elif action_type == ActionType.APPLY_LEAVE.value:
                result = self._apply_leave(parameters)
            elif action_type == ActionType.UPDATE_DOCUMENTATION.value:
                result = self._update_documentation(parameters)
            else:
                result = {"status": "error", "message": f"Unknown action type: {action_type}"}
            
            log_entry = {
                "action_id": action_id,
                "action_type": action_type,
                "timestamp": timestamp,
                "parameters": parameters,
                "result": result
            }
            self.action_log.append(log_entry)
            self.logger.info(f"Action executed: {action_type} - {action_id}")
            
            return {
                "action_id": action_id,
                "action_type": action_type,
                "timestamp": timestamp,
                "status": result.get("status", "success"),
                "details": result
            }
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return {
                "action_id": action_id,
                "action_type": action_type,
                "timestamp": timestamp,
                "status": "error",
                "error": str(e)
            }
    
    def _file_ticket(self, params: Dict) -> Dict:
        ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "ticket_id": ticket_id,
            "title": params.get("title", "Support Request"),
            "description": params.get("description", ""),
            "priority": params.get("priority", "medium"),
            "category": params.get("category", "general"),
            "assigned_to": "IT Support Team",
            "expected_resolution": "48 hours",
            "message": f"Ticket {ticket_id} has been created successfully"
        }
    
    def _schedule_meeting(self, params: Dict) -> Dict:
        meeting_id = f"MEET-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "meeting_id": meeting_id,
            "title": params.get("title", "Team Meeting"),
            "attendees": params.get("attendees", []),
            "date": params.get("date", "TBD"),
            "time": params.get("time", "TBD"),
            "duration": params.get("duration", "30 minutes"),
            "location": params.get("location", "Virtual"),
            "message": f"Meeting {meeting_id} scheduled successfully"
        }
    
    def _request_software(self, params: Dict) -> Dict:
        request_id = f"SOFT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "request_id": request_id,
            "software_name": params.get("software_name", ""),
            "version": params.get("version", "latest"),
            "justification": params.get("justification", ""),
            "approval_required": True,
            "estimated_approval_time": "3-5 business days",
            "message": f"Software request {request_id} submitted for approval"
        }
    
    def _escalate_issue(self, params: Dict) -> Dict:
        escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "escalation_id": escalation_id,
            "original_ticket": params.get("ticket_id", ""),
            "escalated_to": params.get("escalate_to", "Senior Support Team"),
            "reason": params.get("reason", ""),
            "priority": "high",
            "message": f"Issue escalated successfully with ID {escalation_id}"
        }
    
    def _apply_leave(self, params: Dict) -> Dict:
        leave_id = f"LEAVE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "leave_id": leave_id,
            "leave_type": params.get("leave_type", "casual"),
            "start_date": params.get("start_date", ""),
            "end_date": params.get("end_date", ""),
            "days": params.get("days", 0),
            "reason": params.get("reason", ""),
            "approval_status": "pending",
            "approver": "Manager",
            "message": f"Leave application {leave_id} submitted for approval"
        }
    
    def _update_documentation(self, params: Dict) -> Dict:
        update_id = f"DOC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return {
            "status": "success",
            "update_id": update_id,
            "document": params.get("document_name", ""),
            "section": params.get("section", ""),
            "changes": params.get("changes", ""),
            "reviewer": "Documentation Team",
            "message": f"Documentation update {update_id} queued for review"
        }
    
    def get_action_history(self) -> List[Dict]:
        return self.action_log
    
    def clear_history(self):
        self.action_log = []
        self.logger.info("Action history cleared")


def extract_action_parameters(query: str, action_type: str) -> Dict[str, Any]:
    params = {}
    query_lower = query.lower()
    
    if action_type == ActionType.FILE_TICKET.value:
        params = {
            "title": query[:100],
            "description": query,
            "priority": "high" if "urgent" in query_lower else "medium",
            "category": "IT Support"
        }
    elif action_type == ActionType.SCHEDULE_MEETING.value:
        params = {"title": "Meeting Request", "attendees": [], "duration": "30 minutes"}
    elif action_type == ActionType.REQUEST_SOFTWARE.value:
        params = {"software_name": "Requested Software", "justification": query}
    elif action_type == ActionType.APPLY_LEAVE.value:
        params = {"leave_type": "casual", "reason": query}
    elif action_type == ActionType.ESCALATE_ISSUE.value:
        params = {"reason": query, "escalate_to": "Senior Support Team"}
    elif action_type == ActionType.UPDATE_DOCUMENTATION.value:
        params = {"document_name": "Unknown", "changes": query}
    
    return params