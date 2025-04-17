from qiskit_ibm_runtime import QiskitRuntimeService


service = QiskitRuntimeService()
all_jobs = service.jobs()
pending_jobs = [job for job in all_jobs if job.status() == "QUEUED"]
cancelled_jobs = [job for job in all_jobs if job.status() == "CANCELLED"]

if pending_jobs:
    for job in pending_jobs:
        print(f"Job ID: {job.job_id()}")
        job.cancel()

if cancelled_jobs:
    print("Cancelled jobs IDs:")
    for job in cancelled_jobs:
        print(f"{job.job_id()}")

