from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize the service
service = QiskitRuntimeService()
# Retrieve all jobs
all_jobs = service.jobs()

# Filter jobs with status "QUEUED"
pending_jobs = [job for job in all_jobs if job.status() == "QUEUED"]
cancelled_jobs = [job for job in all_jobs if job.status() == "CANCELLED"]

# Print job IDs
if pending_jobs:
    for job in pending_jobs:
        print(f"Job ID: {job.job_id()}")
        job.cancel()

if cancelled_jobs:
    print("Cancelled jobs IDs:")
    for job in cancelled_jobs:
        print(f"{job.job_id()}")

