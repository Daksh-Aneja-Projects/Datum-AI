# Terraform configuration for AI-Driven Metrology Manufacturing System

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment (dev/staging/prod)"
  type        = string
  default     = "prod"
}

# VPC Network
resource "google_compute_network" "metrology_vpc" {
  name                    = "metrology-${var.environment}-vpc"
  auto_create_subnetworks = false
}

# Subnets
resource "google_compute_subnetwork" "metrology_subnet" {
  name          = "metrology-${var.environment}-subnet"
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  network       = google_compute_network.metrology_vpc.id
}

# GKE Cluster
resource "google_container_cluster" "metrology_cluster" {
  name     = "metrology-${var.environment}-cluster"
  location = var.region
  
  initial_node_count = 3
  
  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
  
  network    = google_compute_network.metrology_vpc.name
  subnetwork = google_compute_subnetwork.metrology_subnet.name
}

# Cloud Storage Buckets
resource "google_storage_bucket" "data_lake" {
  name     = "metrology-${var.environment}-data-lake"
  location = var.region
  
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket" "model_registry" {
  name     = "metrology-${var.environment}-model-registry"
  location = var.region
  
  uniform_bucket_level_access = true
}

# BigQuery Dataset
resource "google_bigquery_dataset" "manufacturing_data" {
  dataset_id = "manufacturing_${var.environment}"
  location   = var.region
  
  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }
}

# Pub/Sub Topics
resource "google_pubsub_topic" "interferometry_data" {
  name = "interferometry-data-${var.environment}"
}

resource "google_pubsub_topic" "sensor_readings" {
  name = "sensor-readings-${var.environment}"
}

resource "google_pubsub_topic" "quality_metrics" {
  name = "quality-metrics-${var.environment}"
}

# Service Accounts
resource "google_service_account" "data_pipeline_sa" {
  account_id   = "metrology-data-pipeline-${var.environment}"
  display_name = "Data Pipeline Service Account"
}

resource "google_project_iam_member" "data_pipeline_roles" {
  for_each = toset([
    "roles/storage.objectAdmin",
    "roles/bigquery.dataEditor",
    "roles/pubsub.publisher"
  ])
  
  role    = each.key
  member  = "serviceAccount:${google_service_account.data_pipeline_sa.email}"
  project = var.project_id
}

# Outputs
output "cluster_endpoint" {
  value = google_container_cluster.metrology_cluster.endpoint
}

output "cluster_ca_certificate" {
  value = google_container_cluster.metrology_cluster.master_auth[0].cluster_ca_certificate
}

output "data_lake_bucket_url" {
  value = google_storage_bucket.data_lake.url
}

output "model_registry_bucket_url" {
  value = google_storage_bucket.model_registry.url
}