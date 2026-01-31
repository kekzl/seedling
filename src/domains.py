"""
Domain Templates for Seedling.

Pre-defined domains with seed instructions for various technical areas.
Each domain contains a description, list of topics, and seed instructions
that can be used to generate synthetic training data.
"""

from __future__ import annotations

from typing import Any

DOMAIN_TEMPLATES = {
    "DevOps": {
        "description": "Infrastructure as Code, CI/CD, Containerization, Orchestration",
        "topics": [
            "Terraform", "Ansible", "Kubernetes", "Docker", "Helm",
            "GitLab CI", "GitHub Actions", "Jenkins", "ArgoCD",
            "Prometheus", "Grafana", "ELK Stack"
        ],
        "seeds": [
            "Schreibe eine Terraform-Konfiguration für einen AWS S3 Bucket mit Versionierung und Encryption",
            "Erstelle ein Ansible Playbook das Docker auf Ubuntu Servern installiert",
            "Schreibe ein Kubernetes Deployment für eine Node.js Anwendung mit 3 Replicas",
            "Erstelle eine GitHub Actions Pipeline die Tests ausführt und ein Docker Image baut",
            "Schreibe ein Helm Chart für eine Redis Installation mit Persistence",
            "Erstelle eine Docker Compose Datei für einen LAMP Stack",
            "Schreibe einen Prometheus Alert für hohe CPU-Auslastung",
            "Erstelle ein GitLab CI Pipeline das Terraform Pläne automatisch erstellt",
            "Schreibe ein Kubernetes ConfigMap das Environment Variables aus einer .env Datei lädt",
            "Erstelle ein Ansible Role für die Nginx Konfiguration mit SSL",
            "Schreibe einen ArgoCD Application Manifest für eine GitOps Deployment",
            "Erstelle eine Grafana Dashboard JSON für Container Metriken",
            "Schreibe ein Terraform Modul für ein VPC mit öffentlichen und privaten Subnets",
            "Erstelle ein Kubernetes CronJob das täglich Datenbank-Backups erstellt",
            "Schreibe eine Jenkins Pipeline die Multi-Branch Builds unterstützt",
        ]
    },
    
    "SysAdmin": {
        "description": "Windows/Linux Administration, Scripting, Identity Management",
        "topics": [
            "PowerShell", "Bash", "Active Directory", "Entra ID",
            "Intune", "SCCM", "Group Policy", "Linux Administration",
            "Systemd", "Networking", "Storage", "Backup"
        ],
        "seeds": [
            "Schreibe ein PowerShell-Skript das alle deaktivierten User aus Active Directory exportiert",
            "Erstelle ein Bash-Skript das Log-Dateien älter als 30 Tage löscht",
            "Schreibe ein PowerShell-Skript das Microsoft 365 Lizenzen für alle User auflistet",
            "Erstelle ein Bash-Skript das den Festplattenspeicher überwacht und bei 90% eine Email sendet",
            "Schreibe ein PowerShell-Skript das Intune Device Compliance Status abruft",
            "Erstelle eine Group Policy die USB-Speichergeräte blockiert",
            "Schreibe ein Bash-Skript das automatisch SSL-Zertifikate mit Certbot erneuert",
            "Erstelle ein PowerShell-Skript das Exchange Online Mailbox-Größen auflistet",
            "Schreibe ein Systemd Service File für eine Python-Anwendung",
            "Erstelle ein PowerShell-Skript das Azure AD Conditional Access Policies exportiert",
            "Schreibe ein Bash-Skript das einen MySQL Dump erstellt und zu S3 hochlädt",
            "Erstelle ein PowerShell-Skript das Windows Event Logs nach Fehlern durchsucht",
            "Schreibe ein Bash-Skript das Nginx Access Logs analysiert und Top-IPs zeigt",
            "Erstelle ein PowerShell-Skript das Microsoft Graph API nutzt um User zu erstellen",
            "Schreibe ein Bash-Skript das SSH Keys auf mehreren Servern rotiert",
        ]
    },
    
    "Cloud": {
        "description": "AWS, Azure, GCP Cloud Services und Architekturen",
        "topics": [
            "AWS", "Azure", "GCP", "IAM", "Networking",
            "Serverless", "Lambda", "Functions", "Storage",
            "Database", "VPC", "Load Balancing", "CDN"
        ],
        "seeds": [
            "Schreibe eine AWS Lambda Funktion die S3 Events verarbeitet und in DynamoDB speichert",
            "Erstelle eine Azure Function die Blob Storage Uploads verarbeitet",
            "Schreibe eine CloudFormation Template für eine Auto Scaling Group",
            "Erstelle ein GCP Cloud Run Service Manifest",
            "Schreibe eine AWS IAM Policy für minimale S3 Berechtigungen",
            "Erstelle eine Azure Resource Manager Template für ein Virtual Network",
            "Schreibe ein AWS CDK Script für eine REST API mit API Gateway und Lambda",
            "Erstelle eine GCP IAM Binding für Service Account Impersonation",
            "Schreibe eine AWS Step Functions Definition für einen Approval Workflow",
            "Erstelle ein Azure Logic App für Email-zu-Ticket Automation",
            "Schreibe eine CloudWatch Alarm Konfiguration für Lambda Errors",
            "Erstelle eine GCP Cloud Function die Pub/Sub Messages verarbeitet",
            "Schreibe eine AWS EventBridge Rule für scheduled Lambda Invocations",
            "Erstelle eine Azure Key Vault Access Policy",
            "Schreibe ein Terraform Script für AWS RDS mit Multi-AZ",
        ]
    },
    
    "Security": {
        "description": "Information Security, Compliance, Hardening",
        "topics": [
            "ISMS", "ISO 27001", "IT-Grundschutz", "GDPR",
            "Hardening", "Penetration Testing", "Incident Response",
            "Vulnerability Management", "SIEM", "Zero Trust"
        ],
        "seeds": [
            "Schreibe eine Security Policy für die Passwortkomplexität nach ISO 27001",
            "Erstelle eine Checkliste für Linux Server Hardening nach CIS Benchmarks",
            "Schreibe ein PowerShell-Skript das Windows Security Baseline prüft",
            "Erstelle eine Incident Response Procedure für Ransomware-Angriffe",
            "Schreibe ein Bash-Skript das offene Ports scannt und dokumentiert",
            "Erstelle eine GDPR-konforme Datenverarbeitungsvereinbarung Template",
            "Schreibe ein Python-Skript das SSL-Zertifikate auf Ablaufdatum prüft",
            "Erstelle eine Zero Trust Architecture Design Document Outline",
            "Schreibe ein Splunk Query für Failed Login Attempts Detection",
            "Erstelle eine Vulnerability Assessment Report Template",
            "Schreibe ein PowerShell-Skript das Azure AD Sign-in Logs analysiert",
            "Erstelle eine Business Continuity Plan Outline nach ISO 22301",
            "Schreibe ein Bash-Skript das Firewall Rules dokumentiert",
            "Erstelle eine Security Awareness Training Agenda",
            "Schreibe ein Python-Skript das HIBP API für Email-Leak-Checks nutzt",
        ]
    },
    
    "Database": {
        "description": "SQL, NoSQL, Data Engineering, ETL",
        "topics": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis",
            "Elasticsearch", "SQL", "ETL", "Data Modeling",
            "Backup", "Replication", "Performance Tuning"
        ],
        "seeds": [
            "Schreibe eine PostgreSQL Query die die Top 10 langsamen Queries findet",
            "Erstelle ein MongoDB Aggregation Pipeline für User Activity Analytics",
            "Schreibe ein SQL Script das Foreign Key Constraints dokumentiert",
            "Erstelle ein Python ETL Script das CSV in PostgreSQL lädt",
            "Schreibe eine MySQL Stored Procedure für Audit Logging",
            "Erstelle ein Redis Caching Pattern für Session Management",
            "Schreibe ein SQL Query das Datenbank-Größen aller Tabellen zeigt",
            "Erstelle ein PostgreSQL Backup Script mit pg_dump und Rotation",
            "Schreibe ein MongoDB Index Optimization Script",
            "Erstelle ein SQL Migration Script für Schema-Änderungen",
            "Schreibe ein Python Script das Elasticsearch Indices verwaltet",
            "Erstelle eine PostgreSQL Replication Konfiguration für Read Replicas",
            "Schreibe ein SQL Query das Dead Tuples in PostgreSQL findet",
            "Erstelle ein MySQL Performance Schema Query für Lock-Analyse",
            "Schreibe ein Python Script das Database Connections pooled",
        ]
    },
    
    "Code": {
        "description": "Allgemeine Programmierung in verschiedenen Sprachen",
        "topics": [
            "Python", "TypeScript", "JavaScript", "Rust", "Go",
            "REST APIs", "Testing", "Design Patterns", "Refactoring",
            "Documentation", "Code Review"
        ],
        "seeds": [
            "Schreibe eine Python Klasse für ein Repository Pattern mit SQLAlchemy",
            "Erstelle eine TypeScript Utility Funktion für Deep Object Merging",
            "Schreibe einen Go HTTP Handler mit proper Error Handling",
            "Erstelle ein Python Decorator für Retry Logic mit Exponential Backoff",
            "Schreibe eine Rust Funktion für sichere File I/O mit Error Handling",
            "Erstelle ein TypeScript Interface für eine REST API Response",
            "Schreibe ein Python Unit Test für eine async Funktion mit pytest",
            "Erstelle eine Go Struct mit JSON Tags und Validation",
            "Schreibe eine Python Funktion die Environment Variables validiert",
            "Erstelle ein TypeScript Generic für typsichere API Calls",
            "Schreibe ein Python Context Manager für Database Transactions",
            "Erstelle eine Go Middleware für Request Logging",
            "Schreibe ein Python Script das OpenAPI Specs generiert",
            "Erstelle ein TypeScript Enum mit zugehörigen Utility Functions",
            "Schreibe eine Rust CLI Anwendung mit clap für Argument Parsing",
        ]
    },
}


def get_domain_seeds(domain: str) -> list[str]:
    """Get seed instructions for a specific domain."""
    template = DOMAIN_TEMPLATES.get(domain)
    if template:
        return template.get("seeds", [])
    return []


def get_all_topics() -> list[str]:
    """Get all unique topics across all domains."""
    topics = set()
    for template in DOMAIN_TEMPLATES.values():
        topics.update(template.get("topics", []))
    return sorted(list(topics))


def get_domain_description(domain: str) -> str:
    """Get the description for a domain."""
    template = DOMAIN_TEMPLATES.get(domain)
    if template:
        return template.get("description", "")
    return ""
