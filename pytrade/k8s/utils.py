from kubernetes import client, config
from kubernetes.client import V1ConfigMap


def get_config_map(name: str, namespace: str) -> V1ConfigMap:
    config.load_kube_config()
    v1 = client.CoreV1Api()
    res = v1.read_namespaced_config_map(name=name, namespace=namespace)
    return res
