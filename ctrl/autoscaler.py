import subprocess
import time
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import redis
from threading import Thread
import argparse
from kubernetes import client, config
from kubernetes.client import ApiException
import numpy as np


def get_cli():
    """
    Get input arguments from CLI.
    :return:    ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="Autoscaler Command Line Interface")

    parser.add_argument("-m", "--method", type=str,
                        help='The autoscaler (either muOpt, muOpt-H, VPA, or HPA)',
                        choices=["muOpt", "muOpt-H", "VPA", "HPA"],
                        required=True)
    parser.add_argument("-t", "--wctrl", type=int, default=15,
                        help='The control period (default: 15s)', required=False)
    parser.add_argument("-n", "--name", type=str,
                        help='The experiment name', required=True)
    parser.add_argument("-ut", "--utarget", type=float, default=0.5,
                        help='The target utilization (only available for µOpt)', required=False)

    # Parse the command-line arguments
    return parser.parse_args()


class Autoscaler(object):
    method = None
    optProc = None
    name = None
    juliaOptPath = None
    ctrlInterval = None
    lastR = None
    ut = None

    def __init__(self, name, method, juliaOptPath=None, ctrlInterval=None, ut=None):
        self.name = name
        self.method = method
        self.ctrlInterval = ctrlInterval
        self.ut = ut
        if not juliaOptPath.is_file():
            self.logger.error("juliaOptPath does not exist")
            raise ValueError("juliaOptPath does not exist")

        self.juliaOptPath = juliaOptPath
        self.lastR = None

        # Initialization procedures
        self.init_logger()
        self.init_redis()
        self.init_kubernetes()

        # Autoscaler choice
        if self.method == "muOpt":
            self.logger.info("Running the \'muOpt\' autoscaler (in vertical scaling mode).")
            self.start_julia_opt()
        elif self.method == "muOpt-H":
            self.logger.info("Running the \'muOpt\' autoscaler (in horizontal scaling mode).")
            self.start_julia_opt()
        elif self.method == "VPA":
            self.logger.info("Tracking the recommendations from the \'VPA\' autoscaler.")
            self.vpa_thread = Thread(target=self.vpa_tracking)
            self.vpa_thread.start()
        else:
            self.logger.info("\'HPA\' autoscaler selected. Remember to activate it with kubectl.")

        # Start main loop
        self.main_loop()

    def init_kubernetes(self):
        """
        Configure connection to Kubernetes API server and create Kubernetes API clients.
        :return:
        """

        config.load_kube_config()

        self.vpa_api = client.CustomObjectsApi()
        self.core_v1_api = client.CoreV1Api()
        self.apps_v1_api = client.AppsV1Api()

    def init_redis(self):
        """
        Initialize the redis connection and the actuator Thread.
        :return:
        """
        try:
            self.rCon = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.srvPubSub = self.rCon.pubsub()
            self.srvPubSub.psubscribe("%s_srv" % (self.name))
            self.actuator = Thread(target=self.update_all_pods, args=(self.srvPubSub,))
            self.actuator.start()
        except Exception as e:
            self.logger.error("initRedis failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def init_logger(self):
        """
        Initialize the logger.
        :return:
        """
        try:
            Path("logs/%s" % (self.name)).mkdir(parents=True, exist_ok=True)
            log_file = "logs/%s/%s.log" % (self.name, self.name)

            max_file_size_bytes = 512000  # Set the maximum size of each log file (in bytes)
            backup_count = 5  # Set the number of backup log files to keep
            file_handler = RotatingFileHandler(log_file, maxBytes=max_file_size_bytes, backupCount=backup_count,
                                               mode='w')
            formatter = logging.Formatter('%(threadName)s - %(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger = logging.getLogger(self.name)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error("initLogger failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def vpa_tracking(self):
        self.logger.info("Inside vpa_tracking")
        while True:
            reqs = []
            for i in range(1, 4):
                reqs.append(self.get_cpu_str_by_vpa(f"tier{i}-vpa"))
            combined_reqs = "-".join(reqs)
            channel_name = f"{self.name}_srv"
            self.logger.info(f"Publishing {combined_reqs} to channel {channel_name}")
            self.rCon.publish(channel_name, combined_reqs)
            time.sleep(self.ctrlInterval)

    def start_julia_opt(self):
        """
        Start the Julia optimization (muOpt)
        :return:
        """
        try:
            self.optProc = subprocess.Popen(["julia", str(self.juliaOptPath), "--name", self.name,
                                             "--log_path", "logs/%s/%s_opt.log" % (self.name, self.name),
                                             "--ut", str(self.ut)],
                                            stdout=subprocess.DEVNULL)
            p = self.rCon.pubsub()
            p.psubscribe("%s_strt" % (self.name))
            while True:
                self.logger.info("waiting julia to start")
                msg = p.get_message()
                if (msg is not None and msg["channel"] == "%s_strt" % (self.name) and msg["data"] == "started"):
                    print("Julia started")
                    self.logger.info("Julia started")
                    p.unsubscribe("%s_strt" % (self.name))
                    break
                time.sleep(0.5)
        except Exception as e:
            self.logger.error("start_julia_opt failed with full error trace:")
            self.logger.error(e, exc_info=True)
            self.srvPubSub.unsubscribe()
            self.optProc.kill()
            raise

    def main_loop(self):
        try:
            while True:
                self.logger.info("Main Iteration")
                users = max(self.get_users(), 1)
                self.communicate_users(users)
                time.sleep(self.ctrlInterval)
        except Exception as e:
            self.logger.error("main_loop failed with full error trace:")
            self.logger.error(e, exc_info=True)
        finally:
            self.srvPubSub.unsubscribe()
            self.optProc.terminate()

    def get_pod_names_by_deployment(self, deployment_name):
        pods = []
        try:
            all_pods = self.core_v1_api.list_namespaced_pod(namespace='default')
            for pod in all_pods.items:
                pod_name = pod.metadata.name
                if deployment_name in pod_name:
                    pods.append(pod_name)
            return pods
        except client.ApiException as e:
            print(f"Exception when calling CoreV1Api->list_namespaced_pod: {e}\n")

    def get_users(self):
        """
        Retrieve the current number of users from Redis.

        :return:    The current number of users.
        """
        try:
            users = int(self.rCon.get(f"{self.name}_wrk"))
            return users
        except Exception as e:
            self.logger.error("getUsers failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def communicate_users(self, usr):
        try:
            self.logger.info(f"Sending users {usr}")
            self.rCon.publish(f"{self.name}_usr", str(usr))
        except Exception as e:
            self.logger.error("communicate_users failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def update_all_pods(self, pubsub):
        # Horizontal Scaling
        if self.method == "muOpt-H":
            try:
                for m in pubsub.listen():
                    if 'pmessage' != m['type']:
                        continue
                    replicas = m['data'].split("-")

                    if self.lastR is None:
                        self.lastR = {}
                    for idx, r in enumerate(replicas):
                        tier_number = idx + 1
                        self.logger.info(f"Updating tier{tier_number} to {float(r)} replicas")
                        new_replicas = np.ceil(float(r))
                        if f"tier{tier_number}" not in self.lastR:
                            self.logger.info("DEBUG1")
                            self.lastR[f"tier{tier_number}"] = new_replicas
                            self.horizontally_scale_deployment(tier_number, new_replicas)
                        else:
                            self.logger.info("DEBUG2")
                            if self.lastR[f"tier{tier_number}"] > new_replicas:
                                self.logger.info("DEBUG3")
                                self.logger.info(f"Downscaling tier{tier_number} " + str(
                                    self.lastR[f"tier{tier_number}"]) + f"->{new_replicas}")
                                self.horizontally_scale_deployment(tier_number, new_replicas)
                            elif self.lastR[f"tier{tier_number}"] < new_replicas:
                                self.logger.info("DEBUG4")
                                self.logger.info(
                                    f"Upscaling tier{tier_number} " + str(
                                        self.lastR[f"tier{tier_number}"]) + f"->{float(r)}")
                                self.horizontally_scale_deployment(tier_number, new_replicas)
                            self.lastR[f"tier{tier_number}"] = new_replicas
                            self.logger.info("DEBUG5")
            except Exception as e:
                self.logger.error("mainLoop failed with full error trace:")
                self.logger.error(e, exc_info=True)
        else:  # Vertical Scaling
            try:
                for m in pubsub.listen():
                    if 'pmessage' != m['type']:
                        continue
                    requests = m['data'].split("-")

                    if self.lastR is None:
                        self.lastR = {}
                    for idx, request in enumerate(requests):
                        tier_number = idx + 1
                        deployment_name = f"spring-test-app-tier{tier_number}"
                        container_name = f"{deployment_name}-container"
                        cpu_request = f"{int(float(request) * 1000)}m"
                        cpu_limit = f"{int(float(request) * 1100)}m"
                        self.logger.info(
                            f"Updating tier{tier_number} to CPU request {cpu_request} and CPU limit {cpu_limit}")

                        pod_names = self.get_pod_names_by_deployment(deployment_name)

                        for pod_name in pod_names:
                            self.vertically_scale_pod(pod_name, container_name, cpu_request, cpu_limit)

            except Exception as e:
                self.logger.error("update_all_pods failed with full error trace:")
                self.logger.error(e, exc_info=True)

    def vertically_scale_pod(self, pod_name, container_name, cpu_request, cpu_limit):
        """
        Vertically scale a pod.

        :param pod_name:        The name of the pod.
        :param container_name:  The name of the container.
        :param cpu_request:     The CPU requests to be set for the pod.
        :param cpu_limit:       The CPU limit to be set for the pod.
        :return:
        """
        patch_body = {
            "spec": {
                "containers": [
                    {
                        "name": container_name,
                        "resources": {
                            "requests": {
                                "cpu": cpu_request
                            },
                            "limits": {
                                "cpu": cpu_limit
                            }
                        }
                    }
                ]
            }
        }
        try:
            self.logger.info(f"Updating pod {pod_name} to CPU request {cpu_request} and CPU limit {cpu_limit}")
            self.core_v1_api.patch_namespaced_pod(name=pod_name, namespace="default", body=patch_body)
        except ApiException as e:
            if e.status == 403:
                print(f"Insufficient permissions to access pod '{pod_name}'.")
            elif e.status == 404:
                print(f"Pod '{pod_name}' not found in namespace 'default'.")
            else:
                print(f"Failed to scale pod: {e}")

    def horizontally_scale_deployment(self, tier, replicas):
        """
        Scale a given tier to a provided target number of replicas.

        :param tier:        The tier to scale.
        :param replicas:    The target number of replicas for the given tier.
        :return:
        """
        deployment_name = f"spring-test-app-tier{tier}"
        deployment = self.apps_v1_api.read_namespaced_deployment(name=deployment_name, namespace='default')

        # Update and patch the deployment spec with desired replicas
        deployment.spec.replicas = replicas
        self.apps_v1_api.patch_namespaced_deployment(name=deployment_name, namespace='default', body=deployment)

        self.logger.info(f"Deployment '{deployment_name}' scaled to {deployment.spec.replicas} replicas.")
        return

    def get_cpu_str_by_vpa(self, vpa_name):
        try:
            api_response = self.vpa_api.list_namespaced_custom_object(group="autoscaling.k8s.io", version="v1",
                                                                      namespace="default",
                                                                      plural="verticalpodautoscalers")

            vpa_data = None
            for vpa in api_response["items"]:
                if vpa["metadata"]["name"] == vpa_name:
                    vpa_data = vpa
                    break
            if vpa_data:
                # Extract container recommendation (assuming only one container)
                container_recommendation = vpa_data['status']['recommendation']['containerRecommendations'][0]

                # Extract CPU values
                # cpu_lower_bound = container_recommendation['lowerBound']['cpu']
                cpu_target = container_recommendation['target']['cpu']  # e.g. 1150m
                # cpu_upper_bound = container_recommendation['upperBound']['cpu']
                if len(cpu_target) > 1:
                    cpu_target_value = int(cpu_target[:-1]) / 1000  # e.g. 1.15
                else:
                    cpu_target_value = cpu_target
                # cpu_limit_millicores = int(cpu_target_millicores * 1.1)
                # cpu_limit = f"{cpu_limit_millicores}m"

                # Print results
                print(f"VPA: {vpa_data['metadata']['name']}")
                # print(f"  CPU Lower Bound: {cpu_lower_bound}")
                print(f"  CPU Target: {cpu_target}")
                # print(f"  CPU Limit: {cpu_limit}")
                # print(f"  CPU Upper Bound: {cpu_upper_bound}")
                self.logger.info(f"Recommended CPU for {vpa_name}: {cpu_target_value}")

                return str(cpu_target_value)
            else:
                print(f"VPA named {vpa_name} not found in the provided data.")

        except client.ApiException as e:
            print(f"Error retrieving VPA details: {e}")


if __name__ == '__main__':
    args = get_cli()
    ctrl = Autoscaler(name=args.name, method=args.method, juliaOptPath=Path(__file__).parent.joinpath("3tier.jl"),
                      ctrlInterval=args.wctrl, ut=args.utarget)
