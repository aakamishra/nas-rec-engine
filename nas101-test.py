from nasbench import api

dataset_path = '/home/ec2-user/nasbench_full.tfrecord'
# Use nasbench_full.tfrecord for full dataset (run download command above).
nasbench = api.NASBench(dataset_path)
print("done.")

