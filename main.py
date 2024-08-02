import random
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class ResourceType(Enum):
    CPU = 1
    GPU = 2
    FPGA = 3

class InstanceType:
    def __init__(self, id, cpu, gpu, fpga, cost_per_hour):
        self.id = id
        self.resources = {
            ResourceType.CPU: cpu,
            ResourceType.GPU: gpu,
            ResourceType.FPGA: fpga
        }
        self.cost_per_hour = cost_per_hour

class Region:
    def __init__(self, name):
        self.name = name
        self.datacenters = []

    def add_datacenter(self, datacenter):
        self.datacenters.append(datacenter)

class DataCenter:
    def __init__(self, name):
        self.name = name
        self.hosts = []

    def add_host(self, host):
        self.hosts.append(host)

    def get_total_resources(self):
        total_resources = defaultdict(int)
        for host in self.hosts:
            for resource_type, amount in host.resources.items():
                total_resources[resource_type] += amount
        return total_resources

class Host:
    def __init__(self, id, resources):
        self.id = id
        self.resources = resources
        self.vms = []

    def add_vm(self, vm):
        if self.can_host(vm):
            self.vms.append(vm)
            self.allocate_resources(vm)
            return True
        return False

    def can_host(self, vm):
        return all(self.resources[r] >= vm.resources[r] for r in ResourceType)

    def allocate_resources(self, vm):
        for r in ResourceType:
            self.resources[r] -= vm.resources[r]

    def release_resources(self, vm):
        for r in ResourceType:
            self.resources[r] += vm.resources[r]

class VM:
    def __init__(self, id, resources):
        self.id = id
        self.resources = resources
        self.tasks = deque()

    def add_task(self, task):
        self.tasks.append(task)

class Task:
    def __init__(self, id, resource_requirements, length):
        self.id = id
        self.resource_requirements = resource_requirements
        self.length = length
        self.dependencies = []
        self.vm = None  # To track which VM executed this task

    def add_dependency(self, task):
        self.dependencies.append(task)

class Workflow:
    def __init__(self, id):
        self.id = id
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def add_dependency(self, task, dependency):
        task.add_dependency(dependency)

class HeterogeneousCloudSimulator:
    def __init__(self):
        self.regions = []
        self.instance_types = []
        self.workflows = []
        self.vm_id_counter = 1
        self.task_id_counter = 0
        self.workflow_id_counter = 0
        self.execution_times = []
        self.total_cost = 0

    def add_region(self, region):
        self.regions.append(region)

    def add_instance_type(self, instance_type):
        self.instance_types.append(instance_type)

    def create_and_add_workflow(self):
        self.workflow_id_counter += 1
        new_workflow = Workflow(self.workflow_id_counter)
        self.workflows.append(new_workflow)
        return new_workflow

    def create_and_add_task(self, workflow, cpu_req=None, gpu_req=None, fpga_req=None, length=None):
        self.task_id_counter += 1
        if cpu_req is None:
            cpu_req = random.uniform(0.1, 1.0)
        if gpu_req is None:
            gpu_req = random.randint(0, 2)
        if fpga_req is None:
            fpga_req = random.randint(0, 1)
        if length is None:
            length = random.randint(5000, 20000)
        
        task = Task(self.task_id_counter, 
                    {ResourceType.CPU: cpu_req, ResourceType.GPU: gpu_req, ResourceType.FPGA: fpga_req}, 
                    length)
        workflow.add_task(task)
        
        # Add dependencies to some previous tasks
        if len(workflow.tasks) > 1:
            num_dependencies = random.randint(1, min(3, len(workflow.tasks) - 1))
            for _ in range(num_dependencies):
                dependency = random.choice(workflow.tasks[:-1])  # Choose from all tasks except the current one
                workflow.add_dependency(task, dependency)
        
        return task


    def run_simulation(self):
        total_tasks = sum(len(workflow.tasks) for workflow in self.workflows)
        completed_tasks = 0

        while self.workflows:
            for workflow in self.workflows[:]:
                if workflow.tasks:
                    task = workflow.tasks.pop(0)
                    self.allocate_task(task)
                    completed_tasks += 1
                    if completed_tasks % 100 == 0:
                        print(f"Completed {completed_tasks}/{total_tasks} tasks")
                else:
                    self.workflows.remove(workflow)
                    print(f"Workflow {workflow.id} completed.")

            for region in self.regions:
                for datacenter in region.datacenters:
                    for host in datacenter.hosts:
                        for vm in host.vms:
                            if vm.tasks:
                                task = vm.tasks.popleft()
                                execution_time = self.calculate_execution_time(vm, task)
                                self.execution_times.append((task.id, execution_time))
                                task.vm = vm  # Track which VM executed this task
                                print(f"Task {task.id} executed in {execution_time:.2f} seconds on VM {vm.id}")

        print("All tasks completed.")
        print(f"Total cost of execution: ${self.total_cost:.2f}")

    def allocate_task(self, task):
        allocated = False
        for region in self.regions:
            for datacenter in region.datacenters:
                for host in datacenter.hosts:
                    for vm in host.vms:
                        if self.can_allocate_task(vm, task):
                            vm.add_task(task)
                            allocated = True
                            return
        if not allocated:
            self.provision_new_vm(task)

    def provision_new_vm(self, task):
        for region in self.regions:
            for datacenter in region.datacenters:
                for host in datacenter.hosts:
                    suitable_instance = self.choose_instance_type(task)
                    if suitable_instance:
                        new_vm = VM(self.vm_id_counter, suitable_instance.resources)
                        self.vm_id_counter += 1
                        if host.add_vm(new_vm):
                            new_vm.add_task(task)
                            print(f"Provisioned new VM {new_vm.id} of type {suitable_instance.id} on Host {host.id} in Datacenter {datacenter.name} in Region {region.name} for Task {task.id}")
                            return
        print(f"Task {task.id} could not be allocated due to insufficient resources.")

    def choose_instance_type(self, task):
        suitable_instances = [
            instance for instance in self.instance_types
            if all(instance.resources[r] >= task.resource_requirements[r] for r in ResourceType)
        ]
        return min(suitable_instances, key=lambda x: x.cost_per_hour) if suitable_instances else None

    def calculate_execution_time(self, vm, task):
        execution_times = []
        for resource_type, requirement in task.resource_requirements.items():
            if requirement > 0:
                resource_speed = vm.resources[resource_type]
                execution_times.append(task.length / (resource_speed * requirement))
        execution_time = max(execution_times)  # Assuming parallel execution
        
        # Calculate cost
        instance_type = self.get_instance_type(vm)
        cost = (execution_time / 3600) * instance_type.cost_per_hour  # Convert seconds to hours
        self.total_cost += cost
        
        return execution_time

    def get_instance_type(self, vm):
        return next(instance for instance in self.instance_types if instance.resources == vm.resources)

    def can_allocate_task(self, vm, task):
        return all(vm.resources[r] >= task.resource_requirements[r] for r in ResourceType)

def plot_workflow_graph(simulator):
    G = nx.DiGraph()

    # Add nodes for regions, datacenters, hosts, VMs, and tasks
    for region in simulator.regions:
        G.add_node(f"Region_{region.name}", type="region")
        for datacenter in region.datacenters:
            G.add_node(f"DC_{datacenter.name}", type="datacenter")
            G.add_edge(f"Region_{region.name}", f"DC_{datacenter.name}")
            for host in datacenter.hosts:
                G.add_node(f"Host_{host.id}", type="host")
                G.add_edge(f"DC_{datacenter.name}", f"Host_{host.id}")
                for vm in host.vms:
                    G.add_node(f"VM_{vm.id}", type="vm")
                    G.add_edge(f"Host_{host.id}", f"VM_{vm.id}")

    # Add nodes and edges for tasks and their dependencies
    for workflow in simulator.workflows:
        for task in workflow.tasks:
            G.add_node(f"Task_{task.id}", type="task")
            if task.vm:
                G.add_edge(f"VM_{task.vm.id}", f"Task_{task.id}")
            for dep in task.dependencies:
                G.add_edge(f"Task_{dep.id}", f"Task_{task.id}")

    # Set up the layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw the graph
    plt.figure(figsize=(20, 12))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'region'])
    nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=400, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'datacenter'])
    nx.draw_networkx_nodes(G, pos, node_color='yellow', node_size=300, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'host'])
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=200, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'vm'])
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=100, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'task'])
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Cloud Simulation Workflow and Infrastructure")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Ensure you have python-louvain installed


def plot_task_dependencies(workflow):
    G = nx.DiGraph()

    # Add nodes and edges for tasks and their dependencies
    for task in workflow.tasks:
        G.add_node(f"Task_{task.id}", type="task")
        
        for dep in task.dependencies:
            G.add_edge(f"Task_{dep.id}", f"Task_{task.id}")

    # Set up the layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw the graph
    plt.figure(figsize=(20, 12))
    nx.draw_networkx_nodes(G, pos, node_color='red', node_size=100, alpha=0.8, 
                           nodelist=[node for node, data in G.nodes(data=True) if data['type'] == 'task'])
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Task Dependency Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
simulator = HeterogeneousCloudSimulator()

# Define instance types (similar to EC2 instances)
instance_types = [
    InstanceType("t2.micro", cpu=1, gpu=0, fpga=0, cost_per_hour=0.0116),
    InstanceType("c5.large", cpu=2, gpu=0, fpga=0, cost_per_hour=0.085),
    InstanceType("p3.2xlarge", cpu=8, gpu=1, fpga=0, cost_per_hour=3.06),
    InstanceType("f1.2xlarge", cpu=8, gpu=0, fpga=1, cost_per_hour=1.65)
]

for instance_type in instance_types:
    simulator.add_instance_type(instance_type)

# Create regions
regions = ["us-east-1", "us-west-2", "eu-west-1"]
for region_name in regions:
    region = Region(region_name)
    for i in range(2):  # 2 datacenters per region
        dc = DataCenter(f"DC_{i}")
        for j in range(5):  # 5 hosts per datacenter
            host = Host(j, {ResourceType.CPU: 32, ResourceType.GPU: 4, ResourceType.FPGA: 2})
            dc.add_host(host)
        region.add_datacenter(dc)
    simulator.add_region(region)

# Create a single workflow with 1000 tasks and dependencies
workflow = simulator.create_and_add_workflow()
previous_task = None
for i in range(1000):
    cpu_req = random.uniform(0.1, 8.0)
    gpu_req = random.choice([0, 1])
    fpga_req = random.choice([0, 1])
    length = random.randint(5000, 20000)
    task = simulator.create_and_add_task(workflow, cpu_req=cpu_req, gpu_req=gpu_req, fpga_req=fpga_req, length=length)
    
    # Add dependencies
    if previous_task:
        workflow.add_dependency(task, previous_task)
    if i % 10 == 0:  # Create some branching in the workflow
        previous_task = task
    if i % 50 == 0:  # Reset previous_task occasionally to create multiple parallel branches
        previous_task = None

# Plot the task dependency graph before running the simulation
# plot_task_dependencies(workflow)

def plot_task_dependencies_with_communities(workflow):
    G = nx.DiGraph()

    # Add nodes and edges for tasks and their dependencies
    for task in workflow.tasks:
        G.add_node(f"Task_{task.id}", type="task")
        for dep in task.dependencies:
            G.add_edge(f"Task_{dep.id}", f"Task_{task.id}")

    # Convert directed graph to undirected graph for community detection
    undirected_G = G.to_undirected()

    # Community detection using the Louvain method
    partition = community_louvain.best_partition(undirected_G)

    # Set up the layout
    pos = nx.spring_layout(undirected_G, k=0.5, iterations=50)

    # Draw the graph
    plt.figure(figsize=(20, 12))

    # Color nodes by their community
    cmap = plt.get_cmap('viridis')
    colors = [cmap(partition[node]) for node in undirected_G.nodes()]

    nx.draw_networkx_nodes(undirected_G, pos, node_color=colors, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(undirected_G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(undirected_G, pos, font_size=8)

    plt.title("Task Dependency Graph with Communities")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# plot_task_dependencies_with_communities(workflow)

print("Starting simulation...")
simulator.run_simulation()

# Plot the workflow graph after running the simulation
# plot_workflow_graph(simulator)


# Analyze task communities
print("\nAnalyzing task communities...")
# communities, modularity = analyze_task_communities(workflow)

import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain

def plot_workflow_graph_with_communities(simulator):
    G = nx.DiGraph()

    # Add nodes for regions, datacenters, hosts, VMs, and tasks
    for region in simulator.regions:
        G.add_node(f"Region_{region.name}", type="region")
        for datacenter in region.datacenters:
            G.add_node(f"DC_{datacenter.name}", type="datacenter")
            G.add_edge(f"Region_{region.name}", f"DC_{datacenter.name}")
            for host in datacenter.hosts:
                G.add_node(f"Host_{host.id}", type="host")
                G.add_edge(f"DC_{datacenter.name}", f"Host_{host.id}")
                for vm in host.vms:
                    G.add_node(f"VM_{vm.id}", type="vm")
                    G.add_edge(f"Host_{host.id}", f"VM_{vm.id}")

    # Add nodes and edges for tasks and their dependencies
    for workflow in simulator.workflows:
        for task in workflow.tasks:
            G.add_node(f"Task_{task.id}", type="task")
            if task.vm:
                G.add_edge(f"VM_{task.vm.id}", f"Task_{task.id}")
            for dep in task.dependencies:
                G.add_edge(f"Task_{dep.id}", f"Task_{task.id}")

    # Convert the directed graph to an undirected graph
    H = G.to_undirected()

    # Perform community detection
    partition = community_louvain.best_partition(H)

    # Set up the layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Draw the graph
    plt.figure(figsize=(20, 12))
    
    # Draw nodes colored by community
    nx.draw_networkx_nodes(G, pos, node_color=[partition[node] for node in G.nodes()], 
                           cmap=plt.cm.rainbow, node_size=100, alpha=0.8)

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Create a legend for node types
    node_types = set(nx.get_node_attributes(G, 'type').values())
    for i, node_type in enumerate(node_types):
        plt.scatter([], [], c='C'+str(i), label=node_type)
    plt.legend()

    plt.title("Cloud Simulation Workflow and Infrastructure with Communities")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Print community information
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    print(f"Number of communities detected: {len(communities)}")
    for community_id, nodes in communities.items():
        print(f"Community {community_id}: {len(nodes)} nodes")
        print(f"Sample nodes: {', '.join(nodes[:5])}...")
        print()

# Call the function
# plot_workflow_graph_with_communities(simulator)


# Plot the task dependencies


# plot_task_dependencies(workflow)

