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
#import Webapp


acmeair_replacement_dict = {
    "MSauth" : "acmeair-auth",
    "MSvalidateid" : "acmeair-customer-validateid",
    "MSbookflights" : "acmeair-booking-bookflights",
    "MSupdateMiles" : "acmeair-customer-updatemiles",
    "MScancelbooking" : "acmeair-booking-cancelbooking",
    "MSgetrewardmiles" : "acmeair-flight-getrewardmiles", 
    "MSqueryflights" : "acmeair-flight-queryflights",
    "MSviewprofile" : "acmeair-customer-byidget",
    "MSupdateprofile" : "acmeair-customer-byidpost"
}


acmeair_keywords = ["acmeair-main", "acmeair-auth",
                       "acmeair-customer-byidget", "acmeair-customer-byidpost", "acmeair-customer-updatemiles", "acmeair-customer-validateid",
                       "acmeair-booking-bookflights", "acmeair-booking-bybookingnumber", "acmeair-booking-byuser", "acmeair-booking-cancelbooking",
                       "acmeair-flight-getrewardmiles", "acmeair-flight-queryflights"]
acmeair_vpas = ["vpa-main", "vpa-auth",
                "vpa-byidget", "vpa-byidpost", "vpa-updatemiles", "vpa-validateid",
                "vpa-bookflights", "vpa-bybookingnumber", "vpa-byuser", "vpa-cancelbooking",
                "vpa-getrewardmiles", "vpa-queryflights"]

three_tier_keywords = ["spring-test-app-1", "spring-test-app-2", "spring-test-app-3"]
three_tier_vpas = ["tier1-vpa", "tier2-vpa", "tier3-vpa"]

def get_cli():
    """
    Get input arguments from CLI.
    :return:    ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="Autoscaler Command Line Interface")

    parser.add_argument("-n", "--name", type=str,
                        help='The experiment name', required=True)

    parser.add_argument("-m", "--method", type=str,
                        help='The autoscaler (either muOpt, muOpt-H, VPA, or HPA)',
                        choices=["muOpt", "muOpt-H", "VPA", "HPA"],
                        required=True)
    parser.add_argument("-wa", "--webapp", type=str,
                        help='The name of the benchmark application (either 3tier or Acmeair).',
                        choices=["3tier", "Acmeair"],
                        required=True)

    # Optional arguments
    parser.add_argument("-ut", "--utarget", type=float, default=0.5,
                        help='The target utilization (only available for µOpt)', required=False)
    parser.add_argument("-t", "--wctrl", type=int, default=15,
                        help='The control period (default: 15s)', required=False)

    # Parse the command-line arguments
    return parser.parse_args()


class Autoscaler(object):
    method = None
    webapp = None
    opt_proc = None
    name = None
    julia_opt_path = None
    ctrl_interval = None
    last_r = None
    ut = None
    keywords = None
    vpas = None

    def __init__(self, name, method, webapp, julia_opt_path=None, ctrl_interval=None, ut=None):
        self.name = name
        self.ctrl_interval = ctrl_interval
        self.ut = ut
        self.method = method

        # Get the webapp configuration
        self.webapp = webapp
        if self.webapp == "Acmeair":
            self.keywords = acmeair_keywords
            self.vpas = acmeair_vpas
        else:
            self.keywords = three_tier_keywords
            self.vpas = three_tier_vpas


        if not julia_opt_path.is_file():
            self.logger.error("julia_opt_path does not exist")
            raise ValueError("julia_opt_path does not exist")

        self.julia_opt_path = julia_opt_path
        self.last_r = None

        # Initialization procedures
        self.init_logger()
        self.init_kubernetes()
        self.init_redis()

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
            self.logger.info("\'HPA\' autoscaler selected. Remember to activate it with kubectl (no further action needs to be taken by this program).")

        # Start main loop
        self.main_loop()

    def init_kubernetes(self):
        """
        Configure connection to Kubernetes API server and create Kubernetes API clients.
        :return:
        """

        self.logger.info("Initializing kubernetes APIs")

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
            self.logger.error("init_redis failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def init_logger(self):
        """
        Initialize the logger.
        :return:
        """
        try:
            Path(f"logs/{self.name}").mkdir(parents=True, exist_ok=True)
            log_file = f"logs/{self.name}/{self.name}.log"

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
            self.logger.error("init_logger failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def vpa_tracking(self):
        """
        Tracking the recommendations from VPA and publishing them in the Redis channel.
        :return:
        """
        self.logger.info("Inside vpa_tracking")
        while True:
            reqs = []
            for vpa_name in self.vpas:
                reqs.append(self.get_cpu_str_by_vpa(vpa_name))
            combined_reqs = "_".join(reqs)
            channel_name = f"{self.name}_srv"
            self.logger.info(f"Publishing {combined_reqs} to channel {channel_name}")
            self.rCon.publish(channel_name, combined_reqs)
            time.sleep(self.ctrl_interval)

    def start_julia_opt(self):
        """
        Start the Julia optimization (muOpt).
        :return:
        """
        try:
            self.opt_proc = subprocess.Popen(["julia", str(self.julia_opt_path), "--name", self.name,
                                              "--log_path", f"logs/{self.name}/{self.name}_opt.log",
                                              "--ut", str(self.ut)], stdout=subprocess.DEVNULL)
            p = self.rCon.pubsub()
            p.psubscribe(f"{self.name}_strt")
            while True:
                self.logger.info("waiting julia to start")
                msg = p.get_message()
                if msg is not None and msg["channel"] == f"{self.name}_strt" and msg["data"] == "started":
                    print("Julia started")
                    self.logger.info("Julia started")
                    p.unsubscribe(f"{self.name}_strt")
                    break
                time.sleep(0.5)
        except Exception as e:
            self.logger.error("start_julia_opt failed with full error trace:")
            self.logger.error(e, exc_info=True)
            self.srvPubSub.unsubscribe()
            self.opt_proc.kill()
            raise

    def main_loop(self):
        """
        
        :return:
        """
        try:
            while True:
                self.logger.info("Main Iteration")
                users = max(self.get_users(), 1)
                self.set_users(users)
                time.sleep(self.ctrl_interval)
        except Exception as e:
            self.logger.error("main_loop failed with full error trace:")
            self.logger.error(e, exc_info=True)
        finally:
            self.srvPubSub.unsubscribe()
            self.opt_proc.terminate()

    def get_pod_names_by_deployment(self, deployment_name, namespace='default'):
        """
        Get the list of pods relative to a deployment.
        :param deployment_name:     The deployment name.
        :return:                    The list of pod names.
        """
        pods = []
        try:
            all_pods = self.core_v1_api.list_namespaced_pod(namespace=namespace)
            for pod in all_pods.items:
                pod_name = pod.metadata.name
                if deployment_name in pod_name:
                    pods.append(pod_name)
            return pods
        except Exception as e:
            self.logger.error("get_pod_names_by_deployment failed with full error trace:")
            self.logger.error(e, exc_info=True)

    def get_users(self):
        """
        Retrieve the current number of users from Redis.

        :return:    The current number of users.
        """
        try:
            users = self.rCon.get(f"{self.name}_wrk")
            if(users is None or int(users) <=0):
                self.logger.warning(f"{self.name}_wrk not set, falling back to default number of users 1")
                users=1.
            return float(users)
        except Exception as e:
            self.logger.error("get_users failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def set_users(self, usr):
        """
        Publish the number of users in the Redis channel.
        :param usr: Number of users.
        :return:
        """
        try:
            self.logger.info(f"Sending users {usr}")
            self.rCon.publish(f"{self.name}_usr", str(usr))
        except Exception as e:
            self.logger.error("set_users failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def update_all_pods(self, pubsub):
        """

        :param pubsub:
        :return:
        """
        # Horizontal Scaling
        self.logger.info(f"self.method: {self.method}")
        if self.method == "muOpt-H":
            try:
                for m in pubsub.listen():
                    if 'pmessage' != m['type']:
                        continue
                    self.logger.info(m['data'])
                    res_ctrl = m['data'].split("$")
                    ms_list = res_ctrl[0].split(";")
                    ms_list2 = [acmeair_replacement_dict[ms] for ms in ms_list]
                    replicas = res_ctrl[1].split(";")
                    #replicas = m['data'].split("_")
                    if self.last_r is None:
                        self.last_r = {}
                    for idx, ms in enumerate(ms_list2):
                        deployment_name = f"{ms}-deployment"
                        new_replicas = max(1.0, np.round(float(replicas[idx])))
                        self.logger.info(f"Updating deployment {deployment_name} to {new_replicas} replicas")
                        if deployment_name not in self.last_r:
                            self.last_r[deployment_name] = new_replicas
                            self.horizontally_scale_deployment(deployment_name, new_replicas)
                        else:
                            if self.last_r[deployment_name] > new_replicas:
                                self.logger.info(f"Downscaling {deployment_name} " + str(
                                    self.last_r[deployment_name]) + f"->{new_replicas}")
                                self.horizontally_scale_deployment(deployment_name, new_replicas)
                            elif self.last_r[deployment_name] < new_replicas:
                                self.logger.info(
                                    f"Upscaling {deployment_name} " + str(
                                        self.last_r[deployment_name]) + f"->{float(replicas[idx])}")
                                self.horizontally_scale_deployment(deployment_name, new_replicas)
                            self.last_r[deployment_name] = new_replicas
            except Exception as e:
                self.logger.error("main_loop failed with full error trace:")
                self.logger.error(e, exc_info=True)
        else:  # Vertical Scaling
            try:
                for m in pubsub.listen():
                    if 'pmessage' != m['type']:
                        continue
                    requests = m['data'].split("_")

                    if self.last_r is None:
                        self.last_r = {}
                    for idx, request in enumerate(requests):
                        deployment_name = f"{self.keywords[idx]}-deployment"
                        container_name = f"{self.keywords[idx]}-container"
                        cpu_request = f"{int(float(request) * 1000)}m"
                        cpu_limit = f"{int(float(request) * 1100)}m"
                        self.logger.info(
                            f"Updating {deployment_name} to CPU request {cpu_request} and CPU limit {cpu_limit}")

                        pod_names = self.get_pod_names_by_deployment(deployment_name)

                        for pod_name in pod_names:
                            self.vertically_scale_pod(pod_name, container_name, cpu_request, cpu_limit)

            except Exception as e:
                self.logger.error("update_all_pods failed with full error trace:")
                self.logger.error(e, exc_info=True)

    def vertically_scale_pod(self, pod_name, container_name, cpu_request, cpu_limit, namespace='default'):
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
            self.core_v1_api.patch_namespaced_pod(name=pod_name, namespace=namespace, body=patch_body)
        except ApiException as e:
            if e.status == 403:
                print(f"Insufficient permissions to access pod '{pod_name}'.")
            elif e.status == 404:
                print(f"Pod '{pod_name}' not found in namespace 'default'.")
            else:
                print(f"Failed to scale pod: {e}")

    def horizontally_scale_deployment(self, deployment_name, replicas, namespace='default'):
        """
        Scale a given tier to a provided target number of replicas.

        :param tier:        The tier to scale.
        :param replicas:    The target number of replicas for the given tier.
        :return:
        """
        deployment = self.apps_v1_api.read_namespaced_deployment(name=deployment_name, namespace=namespace)

        # Update and patch the deployment spec with desired replicas
        deployment.spec.replicas = replicas
        self.apps_v1_api.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=deployment)

        self.logger.info(f"Deployment '{deployment_name}' scaled to {deployment.spec.replicas} replicas.")
        return

    def get_cpu_str_by_vpa(self, vpa_name, namespace='default'):
        """

        :param vpa_name:
        :return:
        """
        try:
            api_response = self.vpa_api.list_namespaced_custom_object(group="autoscaling.k8s.io", version="v1",
                                                                      namespace=namespace,
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
                cpu_target = container_recommendation['target']['cpu']  # e.g. 1150m
                if len(cpu_target) > 1:
                    cpu_target_value = int(cpu_target[:-1]) / 1000  # e.g. 1.15
                else:
                    cpu_target_value = cpu_target

                self.logger.info(f"Recommended CPU for {vpa_name}: {cpu_target_value}")
                return str(cpu_target_value)
            else:
                print(f"VPA named {vpa_name} not found in the provided data.")

        except Exception as e:
            self.logger.error("get_cpu_str_by_vpa failed with full error trace:")
            self.logger.error(e, exc_info=True)


if __name__ == '__main__':
    args = get_cli()

    if args.webapp == "Acmeair":
        julia_path = "acmeCtrl.jl"
    else:
        julia_path = "3tier.jl"

    ctrl = Autoscaler(name=args.name, method=args.method, webapp=args.webapp, julia_opt_path=Path(__file__).parent.joinpath(julia_path),
                      ctrl_interval=args.wctrl, ut=args.utarget)
