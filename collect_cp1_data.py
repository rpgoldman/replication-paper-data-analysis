#! /usr/bin/env python

# This script from Mark Weston

import os
import pymongo
from xplan_experiment_analysis import read_credentials
from urllib.parse import quote_plus


def main():
    read_credentials()
    SIFT_GA_MONGO_USER = os.getenv("SIFT_GA_MONGO_USER")
    if SIFT_GA_MONGO_USER is None:
        PermissionError(
            "SIFT_GA_MONGO_USER Environment variable must be set directly or by loading credentials.json file.")
    SIFT_GA_MONGO_HOST = os.getenv("SIFT_GA_MONGO_HOST")
    if SIFT_GA_MONGO_HOST is None:
        SIFT_GA_MONGO_HOST = 'catalog.sd2e.org'
    escaped_user = quote_plus(SIFT_GA_MONGO_USER)
    escaped_host = quote_plus(SIFT_GA_MONGO_HOST)
    dbURI = f'mongodb://readonly:{escaped_user}@{escaped_host}:27020/admin?readPreference=primary'
    client = pymongo.MongoClient(dbURI)
    science_table = client.catalog_staging.science_table
    experiment_flow_jobs_view = client.catalog_staging.experiment_flow_jobs_view
    experiment_ids = science_table.find({"experiment_reference": "Yeast-Gates"}).distinct("experiment_id")
    archive_paths = set()
    experiment_data_map = dict()
    for experiment_id in experiment_ids:
        flow_jobs = experiment_flow_jobs_view.find({"_id.experiment_id": experiment_id})
        found = False
        for flow_job in flow_jobs:
            found = True
            jobs = flow_job["jobs"]
            job_index = len(jobs) - 1
            while job_index >= 0:
                job = jobs[job_index]
                if job["settings"]["analytics"] == "on":
                    print("Found job for {}".format(experiment_id))
                    archive_path = job["archive_path"]
                    print("Archive path {}".format(archive_path))
                    archive_paths.add(archive_path)
                    experiment_data_map[experiment_id] = archive_path
                    job_index = -1
                    continue
                else:
                    print("Found job but analytics is not on {}, continuing".format(experiment_id))
                    job_index = job_index - 1
        if not found:
            print("Did not find job(s) for {}".format(experiment_id))
    print(f"Found {len(archive_paths)} archive paths:")
    for x in sorted(archive_paths):
        print(f"\t{x}")
    return experiment_data_map


if __name__ == '__main__':
    main()
