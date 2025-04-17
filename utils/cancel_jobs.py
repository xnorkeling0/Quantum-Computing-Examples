from qiskit_ibm_runtime import QiskitRuntimeService

# Initialize the service
service = QiskitRuntimeService()
# Retrieve all jobs
all_jobs = service.jobs()

# Filter jobs with status "QUEUED"
pending_jobs = [job for job in all_jobs if job.status() == "QUEUED"]

# Print job IDs
for job in pending_jobs:
    print(f"Job ID: {job.job_id()}")
    job.cancel()

