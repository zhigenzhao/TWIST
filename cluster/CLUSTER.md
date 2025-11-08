# Cluster Guide

1. Download IsaacGym from nvidia, put the file at the same dir with `Dockerfile.twist`.
2. Build the twist image, then pack it into apptainer. Upload the packed image as a `.tar` file.
3. Use `collect_gcc_toolchain.sh` to collect Ubuntu22.04 lib and upload the remote.
4. Put all the retargeted robot data at the correct location on cluster.
5. use `push_job.sh` to submit job.
