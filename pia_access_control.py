import logging
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PIAAccessControl:
    """
    Personally Identifiable Access (PIA) Control System
    
    This class implements access control for document retrieval based on user roles and document metadata
    """
    
    def __init__(self):
        """Initialize the access control system with default role permissions"""
        # Define default role hierarchy (from highest to lowest access level)
        self.role_hierarchy = ["Admin", "Analyst", "Viewer"]
        
        # Define default role permissions
        self.role_permissions = {
            "Admin": {
                "can_access_classifications": ["Public", "Internal", "Confidential", "Restricted"],
                "can_access_departments": ["All"],
                "can_upload": True,
                "can_delete": True,
                "can_modify_permissions": True
            },
            "Analyst": {
                "can_access_classifications": ["Public", "Internal", "Confidential"],
                "can_access_departments": ["Own", "Shared"],
                "can_upload": True,
                "can_delete": False,
                "can_modify_permissions": False
            },
            "Viewer": {
                "can_access_classifications": ["Public", "Internal"],
                "can_access_departments": ["Own"],
                "can_upload": False,
                "can_delete": False,
                "can_modify_permissions": False
            }
        }
    
    def has_permission(self, user_role: str, action: str) -> bool:
        """
        Check if a role has permission to perform an action
        
        Args:
            user_role: The role of the user
            action: The action to check permission for (can_upload, can_delete, etc.)
            
        Returns:
            True if the role has permission, False otherwise
        """
        if user_role not in self.role_permissions:
            logger.warning(f"Unknown role: {user_role}")
            return False
            
        if action not in self.role_permissions[user_role]:
            logger.warning(f"Unknown action: {action}")
            return False
            
        return self.role_permissions[user_role][action]
    
    def filter_documents_by_access(
        self, 
        documents: List[Dict[str, Any]], 
        user_role: str,
        user_department: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter documents based on user role and department
        
        Args:
            documents: List of documents with metadata
            user_role: Role of the user
            user_department: Department of the user (optional)
            
        Returns:
            Filtered list of documents
        """
        if user_role not in self.role_permissions:
            logger.warning(f"Unknown role: {user_role}")
            return []
            
        filtered_docs = []
        permissions = self.role_permissions[user_role]
        
        for doc in documents:
            # Skip documents without metadata
            if "metadata" not in doc:
                continue
                
            metadata = doc["metadata"]
            
            # Filter by classification
            doc_classification = metadata.get("classification", "Public")
            if doc_classification not in permissions["can_access_classifications"]:
                continue
                
            # Filter by department
            if "All" not in permissions["can_access_departments"]:
                doc_department = metadata.get("department", "Unknown")
                
                if user_department and user_department != doc_department:
                    # If not user's department, check if it's a shared department
                    if "Shared" in permissions["can_access_departments"] and metadata.get("shared", False):
                        pass  # Allow access to shared documents
                    else:
                        continue  # Skip documents from other departments
            
            filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_role_level(self, role: str) -> int:
        """
        Get the hierarchical level of a role
        
        Args:
            role: The role to get the level for
            
        Returns:
            The level of the role (lower is higher access), or -1 if not found
        """
        try:
            return self.role_hierarchy.index(role)
        except ValueError:
            return -1
    
    def can_access_document(
        self, 
        user_role: str, 
        document_metadata: Dict[str, Any],
        user_department: Optional[str] = None
    ) -> bool:
        """
        Check if a user can access a specific document
        
        Args:
            user_role: The role of the user
            document_metadata: Metadata of the document
            user_department: Department of the user (optional)
            
        Returns:
            True if the user can access the document, False otherwise
        """
        if user_role not in self.role_permissions:
            return False
            
        permissions = self.role_permissions[user_role]
        
        # Check classification access
        doc_classification = document_metadata.get("classification", "Public")
        if doc_classification not in permissions["can_access_classifications"]:
            return False
            
        # Check department access
        if "All" not in permissions["can_access_departments"]:
            doc_department = document_metadata.get("department", "Unknown")
            
            if "Own" in permissions["can_access_departments"] and user_department == doc_department:
                return True
                
            if "Shared" in permissions["can_access_departments"] and document_metadata.get("shared", False):
                return True
                
            if doc_department != user_department:
                return False
        
        return True
    
    def add_custom_role(
        self, 
        role_name: str, 
        permissions: Dict[str, Any], 
        level: Optional[int] = None
    ) -> bool:
        """
        Add a custom role with specified permissions
        
        Args:
            role_name: Name of the new role
            permissions: Dictionary of permissions
            level: Hierarchical level to insert the role (optional)
            
        Returns:
            True if the role was added successfully, False otherwise
        """
        if role_name in self.role_permissions:
            logger.warning(f"Role {role_name} already exists")
            return False
            
        required_keys = ["can_access_classifications", "can_access_departments", 
                          "can_upload", "can_delete", "can_modify_permissions"]
        
        # Check if all required permissions are provided
        for key in required_keys:
            if key not in permissions:
                logger.error(f"Missing required permission key: {key}")
                return False
        
        # Add the new role
        self.role_permissions[role_name] = permissions
        
        # Update role hierarchy
        if level is not None and 0 <= level <= len(self.role_hierarchy):
            self.role_hierarchy.insert(level, role_name)
        else:
            self.role_hierarchy.append(role_name)
            
        logger.info(f"Added custom role: {role_name}")
        return True
