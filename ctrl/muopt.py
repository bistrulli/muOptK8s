import subprocess
import numpy as np
import time
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import redis
from threading import Thread
import sys
import argparse
from kubernetes import client, config
from kubernetes.client import ApiException

# Configure connection to Kubernetes API server
config.load_kube_config()

# Create a Kubernetes API client
# v1_api = client.AppsV1Api()
v1_api = client.CoreV1Api()


def getCli():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="µOpt Command Line Interface")

    parser.add_argument("-t", "--wctrl", type=int, default=15,
                        help='The µOpt control period (default: 15s)', required=False)

    parser.add_argument("-n", "--name", type=str,
                        help='The µOpt experiment name', required=True)

    parser.add_argument("-ut", "--utarget", type=float, default=0.5,
                        help='The µOpt target utilization', required=False)

    # Add optional flag
    # parser.add_argument("-n", "--covariance", action="store_true", help="Output covariance",required=False)
    # parser.add_argument("-r", "--rmoments", action="store_true", help="Option for computing moments with R package. A running R process is required (default: False)",
    #     default=False,required=False)
    # parser.add_argument("-p", "--parallel", action="store_true", help="Option for activationg parallelization (default: False)",
    #     default=False,required=False)

    # Add list of strings
    # parser.add_argument("-v","--vars", nargs="*", default=[],help="List of output variables",required=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    return args


class muOpt(object):
    optProc = None
    name = None
    juliaOptPath = None
    ctrlInterval = None
    lastR = None
    ut = None

    def __init__(self, name, juliaOptPath=None, ctrlInterval=None, ut=None):
        self.name = name
        self.ctrlInterval = ctrlInterval
        self.ut = ut
        if (not juliaOptPath.is_file()):
            self.logger.error("juliaOptPath does not exsist")
            raise ValueError("juliaOptPath does not exsist")

        self.juliaOptPath = juliaOptPath
        self.lastR = None

        self.initLogger()
        self.initRedis()
        self.startJuliaOpt()
        self.mainLoop()

    def initRedis(self):
        try:
            self.rCon = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.srvPubSub = self.rCon.pubsub()
            self.srvPubSub.psubscribe("%s_srv" % (self.name))
            self.actuator = Thread(target=self.updateReplica, args=(self.srvPubSub,))
            self.actuator.start()
        except Exception as e:
            self.logger.error("initRedis failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def initLogger(self):
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

    def getUsers(self):
        try:
            # u=np.random.randint(low=10,high=100)
            u = int(self.rCon.get(f"{self.name}_wrk"))
            return u
        except Exception as e:
            self.logger.error("getUsers failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def comunicateUsers(self, usr):
        try:
            self.logger.info(f"Sending users {usr}")
            self.rCon.publish(f"{self.name}_usr", str(usr))
        except Exception as e:
            self.logger.error("comunicateUsers failed with full error trace:")
            self.logger.error(e, exc_info=True)
            raise

    def startJuliaOpt(self):
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
            self.logger.error("startJuliaOpt failed with full error trace:")
            self.logger.error(e, exc_info=True)
            self.srvPubSub.unsubscribe()
            self.optProc.kill()
            raise

    def mainLoop(self):
        try:
            while True:
                self.logger.info("Main Iteration")
                u = max(self.getUsers(), 1)
                self.comunicateUsers(u)
                time.sleep(self.ctrlInterval)
        except Exception as e:
            self.logger.error("mainLoop failed with full error trace:")
            self.logger.error(e, exc_info=True)
        finally:
            self.srvPubSub.unsubscribe()
            self.optProc.terminate()

    # def actuateKubeCtl(self, tier, R):
    #     self.logger.info("Actuacting")
    #     R = int(R)
    #     kubelog = open(Path(__file__).parent.joinpath("logs").joinpath(self.name).joinpath(f"tier{tier}.log"), "w+")
    #     kubeproc = subprocess.Popen(
    #         ["kubectl", "scale", "deployment", f"spring-test-app-tier{tier}", f"--replicas={R}"],
    #         stdout=kubelog, stderr=kubelog)
    #     kubelog.close()
    #     return kubeproc

    # def actuate_kubernetes_api(self, tier, R):
    #     """
    #     Scales the deployment of a tier to the specified R replicas.
    #     """
    #     deployment_name = f"spring-test-app-tier{tier}"
    #
    #     # Get the deployment object
    #     deployment = v1_api.read_namespaced_deployment(name=deployment_name, namespace='default')
    #
    #     # Update and patch the deployment spec with desired replicas
    #     deployment.spec.replicas = R
    #     v1_api.patch_namespaced_deployment(name=deployment_name, namespace='default', body=deployment)
    #
    #     # Print confirmation message
    #     self.logger.info(f"Deployment '{deployment_name}' scaled to {deployment.spec.replicas} replicas.")
    #
    #     return

    def actuate_kubernetes_api(self, tier, R):
        """
        Scales the deployment of a tier to the specified R replicas.
        """
        # deployment_name = f"spring-test-app-tier{tier}"

        pod_name = f"tier{tier}-pod"
        container_name = f"tier{tier}-container"
        new_cpu_request = f"{str(R * 1000)}m"

        # Define the patch data with the updated CPU request
        patch_body = {
            "spec": {
                "containers": [
                    {
                        "name": container_name,
                        "resources": {
                            "requests": {
                                "cpu": new_cpu_request
                            }
                        }
                    }
                ]
            }
        }

        try:
            # Get the deployment object

            # deployment = v1_api.read_namespaced_deployment(name=deployment_name, namespace='default')

            # Update and patch the deployment spec with desired replicas
            # deployment.spec.replicas = R
            # deployment.spec.template.spec.containers[0].resources.requests['cpu'] = f"{str(R * 1000)}m"

            # v1_api.patch_namespaced_deployment(name=deployment_name, namespace='default', body=deployment)

            v1_api.patch_namespaced_pod(pod_name, "default", patch_body)

            # Check updated value of CPU request
            # container = deployment.spec.template.spec.containers[0]
            # cpu_request = container.resources.requests.get('cpu', None)
            #
            # # Print confirmation message
            # self.logger.info(f"Deployment '{deployment_name}' scaled to {cpu_request} request.")

        except ApiException as e:
            if e.status == 403:
                self.logger.error(f"Insufficient permissions to access pod '{pod_name}'.")
            elif e.status == 404:
                self.logger.error(f"Pod '{pod_name}' not found in namespace 'default'.")
            else:
                self.logger.error(f"Failed to scale deployment: {e}")

        return

    def updateReplica(self, pubsub):
        try:
            for m in pubsub.listen():
                if 'pmessage' != m['type']:
                    continue
                # self.logger.info(f"Updating replicas: {m['data']}")
                replicas = m['data'].split("-")
                # kubeproc = []
                if self.lastR is None:
                    self.lastR = {}
                for idx, r in enumerate(replicas):
                    self.logger.info(f"updating tier{idx + 1} to {float(r) * 1000}m cores")
                    self.actuate_kubernetes_api(idx + 1, float(r))

                    # if (f"tier{idx + 1}" not in self.lastR):
                    #     self.lastR[f"tier{idx + 1}"] = np.ceil(float(r))
                    #     self.actuate_kubernetes_api(idx + 1, np.ceil(float(r)))
                    # else:
                    #     if (self.lastR[f"tier{idx + 1}"] > np.ceil(float(r))):
                    #         self.logger.info(f"Downscaling tier{idx + 1} " + str(
                    #             self.lastR[f"tier{idx + 1}"]) + f"->{np.ceil(float(r))}")
                    #         self.actuate_kubernetes_api(idx + 1, np.ceil(float(r)))
                    #     elif (self.lastR[f"tier{idx + 1}"] < np.ceil(float(r))):
                    #         self.logger.info(
                    #             f"Upscaling tier{idx + 1} " + str(self.lastR[f"tier{idx + 1}"]) + f"->{float(r)}")
                    #         self.actuate_kubernetes_api(idx + 1, np.ceil(float(r)))
                    #     self.lastR[f"tier{idx + 1}"] = np.ceil(float(r))
        except Exception as e:
            self.logger.error("mainLoop failed with full error trace:")
            self.logger.error(e, exc_info=True)


if __name__ == '__main__':
    args = getCli()
    ctrl = muOpt(name=args.name, juliaOptPath=Path(__file__).parent.joinpath("3tier.jl"),
                 ctrlInterval=args.wctrl, ut=args.utarget)
