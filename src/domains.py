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

    "Networking": {
        "description": "Netzwerkadministration, Protokolle, Routing, Firewalls",
        "topics": [
            "TCP/IP", "DNS", "DHCP", "VPN", "Firewall",
            "Load Balancer", "BGP", "VLAN", "SDN", "IPv6",
            "Wireshark", "iptables", "pfSense"
        ],
        "seeds": [
            "Schreibe ein Bash-Skript das alle aktiven Netzwerkverbindungen mit netstat analysiert",
            "Erstelle eine iptables Konfiguration für einen Webserver mit Rate Limiting",
            "Schreibe ein Python-Skript das DNS Records abfragt und dokumentiert",
            "Erstelle eine pfSense Firewall Rule Dokumentation als Vorlage",
            "Schreibe ein PowerShell-Skript das Windows Firewall Rules exportiert",
            "Erstelle ein Bash-Skript das Netzwerk-Latenz zu mehreren Hosts überwacht",
            "Schreibe eine Nginx Konfiguration für TCP Load Balancing",
            "Erstelle ein Python-Skript das Subnetz-Kalkulationen durchführt",
            "Schreibe ein Bash-Skript das VLAN Konfigurationen auf Linux erstellt",
            "Erstelle eine VPN Konfiguration für WireGuard mit mehreren Peers",
            "Schreibe ein Python-Skript das Wireshark PCAP Dateien analysiert",
            "Erstelle ein Ansible Playbook für die Konfiguration von Network Interfaces",
            "Schreibe ein Bash-Skript das IPv6 Adressen für ein Subnetz generiert",
            "Erstelle eine HAProxy Konfiguration für HTTPS Termination",
            "Schreibe ein Python-Skript das SNMP Daten von Switches abruft",
        ]
    },

    "DataScience": {
        "description": "Machine Learning, Datenanalyse, Visualisierung, Statistik",
        "topics": [
            "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
            "Jupyter", "Matplotlib", "Seaborn", "Feature Engineering",
            "Model Training", "Data Cleaning", "Statistical Analysis"
        ],
        "seeds": [
            "Schreibe ein Python-Skript das einen DataFrame mit Pandas bereinigt und Nullwerte behandelt",
            "Erstelle eine Jupyter Notebook Zelle für explorative Datenanalyse mit Visualisierungen",
            "Schreibe eine Python Funktion für Feature Scaling mit Scikit-learn",
            "Erstelle ein Python-Skript das einen Random Forest Classifier trainiert und evaluiert",
            "Schreibe eine Python Funktion für Cross-Validation mit mehreren Metriken",
            "Erstelle ein Matplotlib Dashboard mit mehreren Subplots für Datenvisualisierung",
            "Schreibe ein Python-Skript das Outlier Detection mit IQR durchführt",
            "Erstelle eine Python Klasse für ein Custom Transformer in Scikit-learn Pipelines",
            "Schreibe ein Python-Skript das Time Series Daten mit Rolling Windows analysiert",
            "Erstelle ein PyTorch Neural Network für binäre Klassifikation",
            "Schreibe eine Python Funktion für One-Hot Encoding mit Pandas",
            "Erstelle ein Python-Skript das Hyperparameter Tuning mit GridSearchCV durchführt",
            "Schreibe ein Seaborn Visualization für Correlation Heatmaps",
            "Erstelle ein Python-Skript das Text Daten für NLP vorverarbeitet",
            "Schreibe eine Python Funktion für Train-Test-Split mit Stratification",
        ]
    },

    "Frontend": {
        "description": "Webentwicklung, UI/UX, Frameworks, Accessibility",
        "topics": [
            "React", "Vue", "Angular", "TypeScript", "CSS",
            "Tailwind", "Responsive Design", "Accessibility", "Testing",
            "State Management", "Web Performance", "Progressive Web Apps"
        ],
        "seeds": [
            "Schreibe eine React Komponente mit useState und useEffect für Daten-Fetching",
            "Erstelle eine Vue 3 Composition API Komponente mit reaktiven Daten",
            "Schreibe eine TypeScript React Custom Hook für Form Validation",
            "Erstelle eine CSS Grid Layout für ein responsives Dashboard",
            "Schreibe eine React Komponente mit Tailwind CSS für eine Card-Liste",
            "Erstelle eine Vue Router Konfiguration mit Navigation Guards",
            "Schreibe einen React Context Provider für globales State Management",
            "Erstelle eine Angular Service Klasse für HTTP Requests mit Error Handling",
            "Schreibe eine React Komponente die Accessibility (ARIA) Best Practices folgt",
            "Erstelle ein Jest Test für eine React Komponente mit Testing Library",
            "Schreibe eine CSS Animation für Loading Spinner",
            "Erstelle eine React Lazy Loading Implementation für Code Splitting",
            "Schreibe eine Progressive Web App Manifest Datei mit Service Worker",
            "Erstelle eine Storybook Story für eine Button Komponente",
            "Schreibe eine React Error Boundary Komponente mit Fallback UI",
        ]
    },

    "Mobile": {
        "description": "Mobile App Entwicklung, iOS, Android, Cross-Platform",
        "topics": [
            "React Native", "Flutter", "Swift", "Kotlin", "iOS",
            "Android", "Mobile UI", "Push Notifications", "App Store",
            "Mobile Security", "Offline Storage", "Gestures"
        ],
        "seeds": [
            "Schreibe eine React Native Komponente mit FlatList für infinite Scrolling",
            "Erstelle ein Flutter Widget für eine Custom Bottom Navigation Bar",
            "Schreibe eine Swift Funktion für Core Data Persistence",
            "Erstelle eine Kotlin Klasse für Room Database Entity und DAO",
            "Schreibe eine React Native Custom Hook für AsyncStorage Management",
            "Erstelle ein Flutter BLoC Pattern für State Management",
            "Schreibe eine Swift Funktion für Push Notification Handling",
            "Erstelle eine Kotlin Funktion für Retrofit API Calls mit Coroutines",
            "Schreibe eine React Native Komponente für Gesture Handling mit PanResponder",
            "Erstelle ein Flutter Widget für responsive Layout mit MediaQuery",
            "Schreibe eine Swift Funktion für Keychain Secure Storage",
            "Erstelle eine Kotlin Funktion für biometrische Authentifizierung",
            "Schreibe eine React Native Navigation Setup mit React Navigation",
            "Erstelle ein Flutter Widget für Image Caching und Loading States",
            "Schreibe eine Swift Funktion für Background Task Scheduling",
        ]
    },

    "QA": {
        "description": "Quality Assurance, Testautomatisierung, Performance Testing",
        "topics": [
            "Selenium", "Cypress", "Playwright", "Jest", "pytest",
            "Load Testing", "API Testing", "Test Planning", "BDD",
            "CI/CD Testing", "Mobile Testing", "Accessibility Testing"
        ],
        "seeds": [
            "Schreibe einen Cypress E2E Test für einen Login Flow",
            "Erstelle ein Playwright Test Script für Cross-Browser Testing",
            "Schreibe einen pytest Test mit Fixtures und Parametrization",
            "Erstelle ein Selenium Page Object Model für eine Webanwendung",
            "Schreibe ein k6 Load Test Script für API Performance Testing",
            "Erstelle eine Jest Mock Implementation für externe Services",
            "Schreibe ein Robot Framework Test Case für Keyword-Driven Testing",
            "Erstelle ein Postman Collection Script für API Test Automation",
            "Schreibe einen pytest Test mit Mocking von Database Connections",
            "Erstelle ein Cypress Custom Command für wiederverwendbare Aktionen",
            "Schreibe ein JMeter Test Plan für Stress Testing",
            "Erstelle ein BDD Scenario mit Gherkin Syntax für User Stories",
            "Schreibe einen Playwright Test für Accessibility Checks mit axe-core",
            "Erstelle ein pytest Plugin für Custom Test Reporting",
            "Schreibe ein Selenium Grid Konfiguration für parallele Tests",
        ]
    },

    "MLOps": {
        "description": "Machine Learning Operations, Model Deployment, Feature Stores",
        "topics": [
            "MLflow", "Kubeflow", "Feature Store", "Model Registry",
            "Model Serving", "A/B Testing", "Model Monitoring",
            "Data Versioning", "DVC", "Experiment Tracking"
        ],
        "seeds": [
            "Schreibe ein MLflow Experiment Tracking Script für Modell-Training",
            "Erstelle eine Kubeflow Pipeline für automatisiertes ML Training",
            "Schreibe ein Python-Skript für Feature Store Integration mit Feast",
            "Erstelle ein Docker Container für Model Serving mit FastAPI",
            "Schreibe ein DVC Pipeline für Data Versioning und Reproduzierbarkeit",
            "Erstelle ein Python-Skript für Model Registry mit MLflow",
            "Schreibe ein Kubernetes Deployment für ML Model Inference",
            "Erstelle ein Python-Skript für A/B Testing von ML Modellen",
            "Schreibe ein Monitoring Dashboard für Model Performance Metriken",
            "Erstelle ein Python-Skript für automatisches Model Retraining",
            "Schreibe ein Airflow DAG für ML Pipeline Orchestration",
            "Erstelle ein Python-Skript für Feature Engineering Pipeline",
            "Schreibe ein Terraform Script für SageMaker Endpoint Deployment",
            "Erstelle ein Python-Skript für Data Drift Detection",
            "Schreibe ein GitHub Actions Workflow für ML CI/CD Pipeline",
        ]
    },

    "API": {
        "description": "API Design, REST, GraphQL, gRPC, Dokumentation",
        "topics": [
            "REST", "GraphQL", "gRPC", "OpenAPI", "Swagger",
            "API Gateway", "Rate Limiting", "Authentication",
            "Versioning", "Webhooks", "API Testing"
        ],
        "seeds": [
            "Schreibe eine FastAPI Route mit Pydantic Validation und Response Models",
            "Erstelle ein GraphQL Schema mit Queries und Mutations",
            "Schreibe eine gRPC Service Definition mit Protocol Buffers",
            "Erstelle eine OpenAPI 3.0 Spezifikation für eine User API",
            "Schreibe eine Express.js Middleware für API Rate Limiting",
            "Erstelle ein Python-Skript für JWT Token Validation",
            "Schreibe eine FastAPI Dependency für Database Session Management",
            "Erstelle ein GraphQL Resolver mit DataLoader für N+1 Optimization",
            "Schreibe eine API Versioning Strategie mit URL Prefixes",
            "Erstelle ein Webhook Handler mit Signature Verification",
            "Schreibe eine FastAPI Exception Handler für konsistente Error Responses",
            "Erstelle ein gRPC Client mit Retry Logic und Timeout Handling",
            "Schreibe eine Swagger UI Konfiguration mit Custom Branding",
            "Erstelle ein Python-Skript für API Health Check Endpoints",
            "Schreibe eine GraphQL Subscription für Real-time Updates",
        ]
    },

    "Docker": {
        "description": "Docker Container, Images, Dockerfile, Registry, Networking",
        "topics": [
            "Dockerfile", "Docker Build", "Docker Run", "Docker Images",
            "Docker Volumes", "Docker Networks", "Docker Registry", "Multi-Stage Builds",
            "Container Debugging", "Docker Logs", "Docker Exec", "Docker Prune",
            "Health Checks", "Environment Variables", "Docker Secrets"
        ],
        "seeds": [
            "Schreibe ein Dockerfile für eine Python Flask Anwendung mit Multi-Stage Build",
            "Erstelle einen Docker Befehl der alle gestoppten Container löscht",
            "Schreibe ein Dockerfile für eine Node.js Anwendung mit npm ci und non-root User",
            "Erstelle einen Docker Run Befehl mit Volume Mounts und Port Mappings",
            "Schreibe ein Dockerfile mit Health Check für einen Nginx Webserver",
            "Erstelle einen Docker Befehl der Logs von mehreren Containern gleichzeitig anzeigt",
            "Schreibe ein Dockerfile für eine Go Anwendung mit scratch Base Image",
            "Erstelle einen Docker Build Befehl mit Build Args und Cache Management",
            "Schreibe einen Docker Befehl der in einen laufenden Container eine Shell öffnet",
            "Erstelle ein Dockerfile für eine Java Spring Boot Anwendung mit JRE Runtime",
            "Schreibe einen Docker Befehl der alle ungenutzten Images und Volumes aufräumt",
            "Erstelle einen Docker Network Befehl für ein isoliertes Bridge Netzwerk",
            "Schreibe ein Dockerfile mit ENTRYPOINT und CMD Kombination für flexible Starts",
            "Erstelle einen Docker Befehl der Container Ressourcen-Nutzung überwacht",
            "Schreibe ein Dockerfile für eine Rust Anwendung mit cargo build --release",
        ]
    },

    "DockerCompose": {
        "description": "Docker Compose Multi-Container Orchestrierung, Services, Networks, Volumes",
        "topics": [
            "docker-compose.yml", "Services", "Networks", "Volumes",
            "Environment Files", "Depends On", "Health Checks", "Profiles",
            "Build Context", "Replicas", "Resource Limits", "Secrets",
            "Override Files", "Docker Compose Watch", "Service Discovery"
        ],
        "seeds": [
            "Schreibe eine docker-compose.yml für einen LAMP Stack mit MySQL und phpMyAdmin",
            "Erstelle eine Docker Compose Konfiguration mit Health Checks und Depends On",
            "Schreibe eine docker-compose.yml für eine Microservices Architektur mit API Gateway",
            "Erstelle eine Docker Compose Konfiguration mit Named Volumes für Persistenz",
            "Schreibe eine docker-compose.yml mit Environment File und Variable Substitution",
            "Erstelle eine Docker Compose Konfiguration für ELK Stack mit Elasticsearch und Kibana",
            "Schreibe eine docker-compose.yml mit Build Context und Multi-Stage Dockerfile",
            "Erstelle eine Docker Compose Konfiguration mit Resource Limits für CPU und Memory",
            "Schreibe eine docker-compose.yml für Redis Cluster mit mehreren Replicas",
            "Erstelle eine Docker Compose Konfiguration mit Custom Networks und Aliases",
            "Schreibe eine docker-compose.yml mit Profiles für Dev und Production Umgebungen",
            "Erstelle eine Docker Compose Override Datei für lokale Entwicklung",
            "Schreibe eine docker-compose.yml für PostgreSQL mit automatischem Backup Service",
            "Erstelle eine Docker Compose Konfiguration mit Traefik Reverse Proxy und SSL",
            "Schreibe eine docker-compose.yml mit Docker Compose Watch für Hot Reload",
        ]
    },

    "Bash": {
        "description": "Shell Scripting, Command Line Tools, Text Processing, Automation",
        "topics": [
            "Bash Scripting", "Shell Variables", "Control Flow", "Functions",
            "Pipes", "Redirection", "Process Management", "Text Processing",
            "File Operations", "Error Handling", "Subshells", "Arrays",
            "Parameter Expansion", "Here Documents", "Signal Handling"
        ],
        "seeds": [
            "Schreibe ein Bash-Skript das Dateien rekursiv nach Größe sortiert auflistet",
            "Erstelle ein Bash-Skript mit Funktionen für farbige Terminal-Ausgaben",
            "Schreibe ein Bash-Skript das Argumente parst mit getopts",
            "Erstelle ein Bash-Skript das parallele Downloads mit Background Jobs durchführt",
            "Schreibe ein Bash-Skript das eine Konfigurationsdatei parst und Variablen setzt",
            "Erstelle ein Bash-Skript mit Error Handling und Trap für Cleanup",
            "Schreibe ein Bash-Skript das Log-Dateien rotiert und komprimiert",
            "Erstelle ein Bash-Skript das Prozesse nach Memory-Verbrauch überwacht",
            "Schreibe ein Bash-Skript mit Here Document für Multi-Line SQL Queries",
            "Erstelle ein Bash-Skript das Dateien nach Datum filtert und verschiebt",
            "Schreibe ein Bash-Skript das ein Verzeichnis auf Änderungen überwacht",
            "Erstelle ein Bash-Skript mit Arrays für Batch-Verarbeitung von Dateien",
            "Schreibe ein Bash-Skript das stdin liest und zeilenweise verarbeitet",
            "Erstelle ein Bash-Skript mit Lockfile für Single-Instance Ausführung",
            "Schreibe ein Bash-Skript das Environment Variables aus .env Datei lädt",
        ]
    },

    "SSH": {
        "description": "SSH Verbindungen, Key Management, Tunneling, Remote Execution",
        "topics": [
            "SSH Keys", "SSH Config", "SSH Tunnel", "Port Forwarding",
            "Jump Hosts", "SSH Agent", "SCP", "SFTP", "Remote Commands",
            "Known Hosts", "Host Key Verification", "ProxyJump", "ControlMaster",
            "SSH Escape Sequences", "Key Rotation"
        ],
        "seeds": [
            "Schreibe einen SSH Befehl für Local Port Forwarding zu einer Datenbank",
            "Erstelle eine SSH Config Datei mit mehreren Hosts und ProxyJump",
            "Schreibe einen SSH Befehl der Remote Commands auf mehreren Servern ausführt",
            "Erstelle ein Bash-Skript das SSH Keys generiert und auf Server verteilt",
            "Schreibe einen SSH Tunnel Befehl für Dynamic SOCKS Proxy",
            "Erstelle eine SSH Config mit ControlMaster für Connection Multiplexing",
            "Schreibe einen SCP Befehl für rekursives Kopieren mit Bandbreitenlimit",
            "Erstelle ein Bash-Skript das SSH Agent mit Key Timeout konfiguriert",
            "Schreibe einen SSH Befehl für Reverse Port Forwarding",
            "Erstelle eine SSH Config mit unterschiedlichen Keys pro Host",
            "Schreibe einen SSH Befehl der über Jump Host eine Verbindung aufbaut",
            "Erstelle ein Bash-Skript das SSH Host Keys in known_hosts verwaltet",
            "Schreibe einen SFTP Batch-Befehl für automatisierte Dateitransfers",
            "Erstelle einen SSH Befehl mit Pseudo-Terminal für interaktive Sessions",
            "Schreibe ein Bash-Skript das SSH Verbindungen mit Timeout überwacht",
        ]
    },

    "Curl": {
        "description": "HTTP Requests, API Testing, Downloads, Authentication",
        "topics": [
            "GET Requests", "POST Requests", "Headers", "Authentication",
            "JSON Data", "File Upload", "Cookies", "SSL/TLS", "Timeouts",
            "Retry Logic", "Output Formatting", "Follow Redirects",
            "Rate Limiting", "Proxy", "Debug Mode"
        ],
        "seeds": [
            "Schreibe einen curl Befehl für POST Request mit JSON Body und Headers",
            "Erstelle einen curl Befehl für API Authentication mit Bearer Token",
            "Schreibe einen curl Befehl der Response Headers und Body getrennt anzeigt",
            "Erstelle einen curl Befehl für File Upload mit multipart/form-data",
            "Schreibe einen curl Befehl mit Retry Logic und Exponential Backoff",
            "Erstelle einen curl Befehl für Download mit Fortschrittsanzeige und Resume",
            "Schreibe einen curl Befehl für Basic Authentication mit URL Encoding",
            "Erstelle einen curl Befehl der Cookies speichert und wiederverwendet",
            "Schreibe einen curl Befehl mit Client Certificate für mTLS",
            "Erstelle einen curl Befehl für GraphQL Query mit Variables",
            "Schreibe einen curl Befehl mit Timeout und Connection Retry",
            "Erstelle einen curl Befehl der Response Time und HTTP Status misst",
            "Schreibe einen curl Befehl für PUT Request mit Datei-Inhalt",
            "Erstelle einen curl Befehl mit Proxy und Proxy Authentication",
            "Schreibe einen curl Befehl für Parallel Requests mit xargs",
        ]
    },

    "Ripgrep": {
        "description": "Schnelle Textsuche, Regex Patterns, Code Search, Filtering",
        "topics": [
            "Pattern Matching", "Regex", "File Type Filtering", "Context Lines",
            "Ignore Patterns", "Case Sensitivity", "Word Boundaries", "Multiline",
            "Replace Mode", "JSON Output", "Glob Patterns", "Hidden Files",
            "Binary Files", "Count Mode", "Files With Matches"
        ],
        "seeds": [
            "Schreibe einen rg Befehl der nach einem Pattern in bestimmten Dateitypen sucht",
            "Erstelle einen rg Befehl mit Context Lines vor und nach dem Match",
            "Schreibe einen rg Befehl für Case-Insensitive Suche mit Word Boundaries",
            "Erstelle einen rg Befehl der nur Dateinamen mit Matches auflistet",
            "Schreibe einen rg Befehl mit Regex für IP-Adressen in Log-Dateien",
            "Erstelle einen rg Befehl der bestimmte Verzeichnisse ignoriert",
            "Schreibe einen rg Befehl für Multiline Pattern Matching",
            "Erstelle einen rg Befehl mit JSON Output für weitere Verarbeitung",
            "Schreibe einen rg Befehl der Matches zählt und gruppiert nach Datei",
            "Erstelle einen rg Befehl mit Glob Pattern für spezifische Pfade",
            "Schreibe einen rg Befehl der auch in Hidden Files und Directories sucht",
            "Erstelle einen rg Befehl mit Replace Mode für Text-Substitution",
            "Schreibe einen rg Befehl für Suche nach TODO und FIXME Kommentaren",
            "Erstelle einen rg Befehl mit Type-Add für Custom Dateitypen",
            "Schreibe einen rg Befehl der Regex Groups captured und anzeigt",
        ]
    },

    "Cat": {
        "description": "Datei-Anzeige, Concatenation, Text-Transformation, Pipe-Operationen",
        "topics": [
            "File Display", "Concatenation", "Line Numbers", "Non-Printing Characters",
            "Heredoc", "Pipe Operations", "File Creation", "Tab Display",
            "End of Line Markers", "Squeeze Blank Lines", "Binary View",
            "Multiple Files", "Standard Input", "Reverse Output", "Pagination"
        ],
        "seeds": [
            "Schreibe einen cat Befehl der mehrere Dateien mit Zeilennummern anzeigt",
            "Erstelle einen cat Befehl mit Here Document für Multi-Line Datei-Erstellung",
            "Schreibe einen cat Befehl der nicht-druckbare Zeichen sichtbar macht",
            "Erstelle einen cat Befehl kombiniert mit grep für gefilterte Ausgabe",
            "Schreibe einen cat Befehl der Tabs als ^I und Zeilenenden als $ anzeigt",
            "Erstelle einen cat Befehl für das Zusammenfügen von Split-Dateien",
            "Schreibe einen cat Befehl mit Pipe zu wc für Zeilen-Statistiken",
            "Erstelle einen cat Befehl der aufeinanderfolgende Leerzeilen komprimiert",
            "Schreibe einen cat Befehl kombiniert mit head und tail für Datei-Ausschnitte",
            "Erstelle einen cat Befehl mit tee für gleichzeitiges Anzeigen und Speichern",
            "Schreibe einen cat Befehl für Binary-Dateien mit hexdump Pipe",
            "Erstelle einen cat Befehl der stdin mit Dateien kombiniert",
            "Schreibe einen cat Befehl mit sort und uniq für deduplizierte Ausgabe",
            "Erstelle einen cat Befehl kombiniert mit sed für Text-Transformation",
            "Schreibe einen cat Befehl mit nl für fortlaufende Nummerierung über Dateien",
        ]
    },

    "Files": {
        "description": "Datei-Formate, Metadaten, Encoding, Konvertierung, Analyse",
        "topics": [
            "File Formats", "MIME Types", "Metadata", "Encoding",
            "File Headers", "Magic Bytes", "Checksums", "Compression",
            "Archive Formats", "File Parsing", "Binary Analysis", "Text Encoding",
            "File Validation", "Format Conversion", "File Structure"
        ],
        "seeds": [
            "Schreibe ein Python-Skript das MIME-Type und Magic Bytes einer Datei analysiert",
            "Erstelle ein Bash-Skript das Datei-Metadaten mit exiftool extrahiert und als JSON ausgibt",
            "Schreibe ein Python-Skript das verschiedene Text-Encodings erkennt und konvertiert",
            "Erstelle ein Python-Skript das ZIP-Archive entpackt und die Struktur dokumentiert",
            "Schreibe ein Bash-Skript das Checksummen für Dateien berechnet und verifiziert",
            "Erstelle ein Python-Skript das PDF-Metadaten und Text-Inhalt extrahiert",
            "Schreibe ein Python-Skript das Office-Dokumente parst und Inhalte als Markdown exportiert",
            "Erstelle ein Bash-Skript das Datei-Header analysiert und Format-Validierung durchführt",
            "Schreibe ein Python-Skript das XML und JSON Dateien validiert und transformiert",
            "Erstelle ein Python-Skript das CSV-Dateien mit verschiedenen Delimitern erkennt und parst",
            "Schreibe ein Bash-Skript das komprimierte Dateien erkennt und optimal dekomprimiert",
            "Erstelle ein Python-Skript das Binärdateien hexadezimal analysiert und Patterns findet",
            "Schreibe ein Python-Skript das Datei-Integrität prüft und Korruption erkennt",
            "Erstelle ein Bash-Skript das Datei-Duplikate basierend auf Content-Hash findet",
            "Schreibe ein Python-Skript das embedded Dateien aus Containern extrahiert",
        ]
    },

    "Images": {
        "description": "Bildverarbeitung, Analyse, OCR, Metadaten, Format-Konvertierung",
        "topics": [
            "Image Formats", "EXIF Metadata", "OCR", "Image Analysis",
            "Color Spaces", "Resolution", "Compression", "Thumbnails",
            "Image Recognition", "Face Detection", "Object Detection", "Image Comparison",
            "Watermarks", "Steganography", "Image Optimization"
        ],
        "seeds": [
            "Schreibe ein Python-Skript das OCR mit Tesseract durchführt und Text aus Bildern extrahiert",
            "Erstelle ein Python-Skript das EXIF-Metadaten analysiert und GPS-Koordinaten extrahiert",
            "Schreibe ein Bash-Skript das Bilder mit ImageMagick batch-konvertiert und optimiert",
            "Erstelle ein Python-Skript das Bildinhalte mit OpenCV analysiert und Objekte erkennt",
            "Schreibe ein Python-Skript das Screenshots analysiert und UI-Elemente identifiziert",
            "Erstelle ein Python-Skript das Bilder auf Duplikate mit Perceptual Hashing vergleicht",
            "Schreibe ein Bash-Skript das Thumbnails in verschiedenen Größen generiert",
            "Erstelle ein Python-Skript das Farbpaletten aus Bildern extrahiert und analysiert",
            "Schreibe ein Python-Skript das Diagramme und Charts in Bildern erkennt und Daten extrahiert",
            "Erstelle ein Python-Skript das Bildqualität bewertet und Kompressionsartefakte erkennt",
            "Schreibe ein Python-Skript das Gesichter in Bildern erkennt und anonymisiert",
            "Erstelle ein Bash-Skript das Bildformate validiert und korrupte Dateien identifiziert",
            "Schreibe ein Python-Skript das Text-Overlays und Wasserzeichen in Bildern analysiert",
            "Erstelle ein Python-Skript das Bild-zu-Text Beschreibungen mit Vision API generiert",
            "Schreibe ein Python-Skript das QR-Codes und Barcodes in Bildern dekodiert",
        ]
    },

    "Audio": {
        "description": "Audio-Verarbeitung, Transkription, Analyse, Format-Konvertierung",
        "topics": [
            "Audio Formats", "Transcription", "Speech Recognition", "Audio Metadata",
            "Waveform Analysis", "Spectrograms", "Noise Reduction", "Audio Compression",
            "Voice Detection", "Music Analysis", "Audio Segmentation", "Pitch Detection",
            "Sample Rate", "Bit Depth", "Audio Normalization"
        ],
        "seeds": [
            "Schreibe ein Python-Skript das Audio-Transkription mit Whisper durchführt",
            "Erstelle ein Python-Skript das Audio-Metadaten mit mutagen extrahiert und bearbeitet",
            "Schreibe ein Bash-Skript das Audio-Dateien mit ffmpeg batch-konvertiert",
            "Erstelle ein Python-Skript das Sprachaktivität in Audio-Dateien erkennt und segmentiert",
            "Schreibe ein Python-Skript das Spektrogramme generiert und Audio visuell analysiert",
            "Erstelle ein Python-Skript das Lautstärke normalisiert und Audio-Qualität verbessert",
            "Schreibe ein Python-Skript das Sprecher-Diarisierung durchführt und Sprecher trennt",
            "Erstelle ein Bash-Skript das Audio-Streams aus Video-Dateien extrahiert",
            "Schreibe ein Python-Skript das Musik-Tempo und Beats analysiert",
            "Erstelle ein Python-Skript das Audio-Dateien auf Stille prüft und splittet",
            "Schreibe ein Python-Skript das Hintergrundgeräusche erkennt und klassifiziert",
            "Erstelle ein Python-Skript das Untertitel aus Audio generiert mit Timestamps",
            "Schreibe ein Bash-Skript das Audio-Qualität analysiert und Bitrate empfiehlt",
            "Erstelle ein Python-Skript das Audio-Fingerprinting für Duplikat-Erkennung durchführt",
            "Schreibe ein Python-Skript das Emotionen und Sentiment in Sprache analysiert",
        ]
    },

    "Video": {
        "description": "Video-Verarbeitung, Analyse, Transkription, Frame-Extraktion",
        "topics": [
            "Video Formats", "Codecs", "Frame Extraction", "Video Metadata",
            "Transcoding", "Scene Detection", "Object Tracking", "Video OCR",
            "Subtitle Extraction", "Video Thumbnails", "Motion Detection", "Video Quality",
            "Streaming Formats", "Video Segmentation", "Content Analysis"
        ],
        "seeds": [
            "Schreibe ein Python-Skript das Keyframes aus Videos extrahiert und analysiert",
            "Erstelle ein Bash-Skript das Videos mit ffmpeg transcodiert und komprimiert",
            "Schreibe ein Python-Skript das Szenen-Wechsel erkennt und Video segmentiert",
            "Erstelle ein Python-Skript das Untertitel aus Videos extrahiert oder generiert",
            "Schreibe ein Python-Skript das Video-Metadaten ausliest und Codec-Info anzeigt",
            "Erstelle ein Python-Skript das Thumbnails zu bestimmten Zeitpunkten generiert",
            "Schreibe ein Bash-Skript das Videos in Streaming-Formate wie HLS konvertiert",
            "Erstelle ein Python-Skript das Bewegung in Videos erkennt und Timestamps markiert",
            "Schreibe ein Python-Skript das Text und Overlays in Video-Frames per OCR extrahiert",
            "Erstelle ein Python-Skript das Video-Qualität analysiert und Encoding empfiehlt",
            "Schreibe ein Python-Skript das Audio-Spur transkribiert und mit Video synchronisiert",
            "Erstelle ein Bash-Skript das Video-Clips zusammenfügt und Übergänge einfügt",
            "Schreibe ein Python-Skript das Gesichter in Videos trackt und Timecodes ausgibt",
            "Erstelle ein Python-Skript das Video-Inhalt beschreibt mit Vision-Language Models",
            "Schreibe ein Python-Skript das Video-Duplikate basierend auf Visual Hashing erkennt",
        ]
    },

    "UV": {
        "description": "Modernes Python Package Management mit uv, Containerisierung, Dependency Resolution",
        "topics": [
            "uv", "pip", "pyproject.toml", "Dependency Management", "Virtual Environments",
            "Lock Files", "Container Python", "Multi-Stage Builds", "Package Publishing",
            "Development Dependencies", "Reproducible Builds", "Cache Optimization",
            "Python Version Management", "Workspace Management", "Tool Installation"
        ],
        "seeds": [
            "Schreibe ein Dockerfile das uv für schnelle Python Dependency Installation nutzt",
            "Erstelle eine pyproject.toml mit uv für ein modernes Python Projekt",
            "Schreibe ein Multi-Stage Dockerfile das uv sync für minimale Container-Größe nutzt",
            "Erstelle ein Bash-Skript das uv für Development Environment Setup verwendet",
            "Schreibe ein Dockerfile das uv Cache optimal für Docker Layer Caching nutzt",
            "Erstelle eine GitHub Actions Pipeline die uv für schnelle CI/CD Builds verwendet",
            "Schreibe ein uv Kommando das Production Dependencies ohne Dev-Dependencies installiert",
            "Erstelle ein Dockerfile das Python Version mit uv python pinnt und installiert",
            "Schreibe ein Bash-Skript das uv lock für reproducible Dependency Resolution nutzt",
            "Erstelle ein Dockerfile das uv tool install für CLI Tools in Container verwendet",
            "Schreibe ein pyproject.toml mit optional Dependencies und Dependency Groups",
            "Erstelle ein Multi-Stage Dockerfile das Build-Dependencies von Runtime trennt mit uv",
            "Schreibe ein Bash-Skript das uv workspace für Monorepo Python Projekte konfiguriert",
            "Erstelle ein Dockerfile das --system Flag für Container ohne venv nutzt",
            "Schreibe ein CI Script das uv cache zwischen Builds persistent speichert",
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
