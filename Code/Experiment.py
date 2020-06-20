import numpy as np
from anytree import AnyNode,Node, RenderTree
import random
import sys
import math
import pandas as pd




def setup_test(w,l):
    # The weights file is expected to be a csv file containing 2 columns. The first row are columns numbers.
    # Each row after that contained an example number followed by its weights. Rows are ordered by example index.
    weights = np.genfromtxt(w, delimiter=',')

    # The links/tree file is expected to be in the format produced by the skicy linkage method.
    # Specificly, each row contain 5 entries: row number, first cluster to be marged, second cluster to be marged, distance between clusters, number of examples in new cluster.
    # Cluster with index under the size of dataset represent leafs containing the example of the same index. Otherwise, the cluster is described in the row numbered: index - number of examples in dataset +1

    links = np.genfromtxt(l, delimiter=',')

    # normally the weights/links csv comes with line and columns numbering, that should be removed
    weights = weights[1:, 1:]
    links = links[1:, 1:]

    size = np.size(weights, 0)  # number of examples in input data
    nodes = np.empty((size + np.size(links, 0)), dtype=AnyNode)

    # creating the leafs + updated weight
    for a in range(size):
        nodes[a] = AnyNode(id="x" + str(a), weight=weights[a,0], num_examples=1, approx_discrepancy=0, Tdiscrepancy=0, delta=0, num_sampled=0)

    # worst_q will calculate the split quality of the hierarchical tree
    worst_q = 0
# creating the internal tree nodes
    for j in range(np.size(links, 0)):
        sum_of_weights = nodes[int(links[j, 0])].weight + nodes[int(links[j, 1])].weight
        sum_of_examples = nodes[int(links[j, 0])].num_examples + nodes[int(links[j, 1])].num_examples
        nodes[size + j] = AnyNode(id="v" + str(j), num_examples=sum_of_examples,
                                        num_sampled=0,  messeured = 0 ,  messeuredBx = 0  ,messeuredBy = 0  ,  weight=sum_of_weights,
                                        approx_discrepancy=0, delta=np.inf, Tdiscrepancy=0)

        nodes[int(links[j, 0])].parent = nodes[size + j]
        nodes[int(links[j, 1])].parent = nodes[size + j]
        # calculating true discrepancy
        leafs = get_leaves(nodes[size + j])
        sum_dis = 0
        for k in range(np.size(leafs, 0)):
            sum_dis = sum_dis + abs(leafs[k].weight - (nodes[size + j].weight / nodes[size + j].num_examples))
        nodes[size + j].Tdiscrepancy = sum_dis
        # calculating q
        if sum_dis > 0:
            if  max(nodes[int(links[j, 0])].Tdiscrepancy , nodes[int(links[j, 1])].Tdiscrepancy) / nodes[size + j].Tdiscrepancy > worst_q:
                worst_q = max(nodes[int(links[j, 0])].Tdiscrepancy , nodes[int(links[j, 1])].Tdiscrepancy) / nodes[size + j].Tdiscrepancy

    # returns the root of the tree
    # print("Split quality q of this tree is: " + str(worst_q))
    return nodes[-1]


# Returns all examples in the subtree rooted at v
def get_leaves(v):
    if v.is_leaf:
        return [v]
    mid_nodes = list(v.children)
    leaf_nodes = []
    while len(mid_nodes) > 0:
        if not mid_nodes[0].is_leaf:
            mid_nodes = mid_nodes + list(mid_nodes[0].children)
            mid_nodes.pop(0)
        else:
            leaf_nodes = leaf_nodes + [mid_nodes.pop(0)]
    return leaf_nodes

# selects node to be sampled via LUCB method
def select_nodes_LUCB(P):
    m = 0
    h_t = P[0]
    for i in range(np.size(P)):
        if P[i].approx_discrepancy > m:
            m = P[i].approx_discrepancy
            h_t = P[i]
    m = 0
    l_t = P[0]
    for i in range(np.size(P)):
        if P[i].approx_discrepancy + P[i].delta >= m:
            if P[i].id != h_t.id:
                m = P[i].approx_discrepancy + P[i].delta
                l_t = P[i]
    if h_t.id == l_t.id:
        if np.size(P) > 1:
            print("PROBLEM IN LUCB SELECT")
            print(P)
    return [h_t, l_t]

# selects node to be sampled via UCB method
def select_nodes_UCB(P):
    m = 0
    l_t = P[0]
    for i in range(np.size(P)):
        if P[i].approx_discrepancy + P[i].delta >= m:
            if P[i].num_sampled < P[i].num_examples:
                m = P[i].approx_discrepancy + P[i].delta
                l_t = P[i]
    # check for extreme cases
    if m == 0:
        print("Reached discrepancy 0")
        print("Pruning size of " + str(np.size(P,0)))
        # return 4
    return [l_t]

# randomly selects an example in node v to query about its weight
def query_node(v):
    leafs = get_leaves(v)
    x = random.choice(leafs)
    x.num_sampled += 1
    v.num_sampled += 1

    v.messeured += abs(x.weight - v.weight / v.num_examples) - x.weight # used for the discrepancy estimator
    v.messeuredBy += pow(abs(x.weight - v.weight / v.num_examples) - x.weight ,2) # used to calculate empirical bernstein based bound

    # if its the first time we query this example, we update the budget accordingly
    if x.num_sampled >= 2:
        return 0
    else:
        return 1

# update estimator and bounds
def update_approxdis_and_delta_Berstein(v, K):
    if not v.is_leaf:
        leafs = get_leaves(v)
        if len(list(filter(lambda a: a.num_sampled == 0, leafs))) == 0:
            v.approx_discrepancy = v.Tdiscrepancy
            v.delta = 0
        elif v.num_sampled > 1:
            v.approx_discrepancy = v.weight + (v.num_examples / v.num_sampled) * v.messeured
        # Delta from paper
            denominator = pow(v.num_sampled*3.2, 2) * K
            small_delta = 3 * Delta / denominator
        # Calculate V_n(m) as in empirical bernstein - faster way
            v_m = (v.num_sampled * v.messeuredBy - pow(v.messeured,2)) / (v.num_sampled * (v.num_sampled - 1))
        # At times due, to numerical computation issues, v_m may result it being negative which is not possible
            if v_m <= 0:
                global Worse_vm
                if v_m < Worse_vm:
                    Worse_vm = v_m
                v_m = 0

        # minimum between empirical berstein and hoeffding
            h = v.weight * (math.sqrt((2 * np.log(4 / small_delta)) / (v.num_sampled)) + (2 * np.log(4 / small_delta)) / (3 * v.num_sampled))
            b = 2 * (v.num_examples) * math.sqrt(2* v_m * np.log(2 / small_delta) / v.num_sampled) + 28 * np.log(2 / small_delta) / (3* (v.num_sampled - 1))
            v.delta = min(h,b)
        # check for instances the bound is false
            if abs(v.approx_discrepancy - v.Tdiscrepancy) > v.delta:
                print("**** BOUND ERROR *****")
                print(v)
                print("**********************")

# calculate discrepancy with regard to w_p
def calculate_dis(P):
    total_dis = 0
    num_leafs = 0
    for i in range(np.size(P)):
        total_dis = total_dis + P[i].Tdiscrepancy
        if P[i].is_leaf:
            num_leafs += 1
    return total_dis


# calculate discrepancy with regard to w'_p
def calculate_dis_modified(P):
    total_dis=0
    for v in P:
        sum_weighted = 0
        sampled = 0
        for x in get_leaves(v):
            if x.num_sampled >= 1:
                sum_weighted += x.weight
                sampled += 1
        for x in get_leaves(v):
            if x.num_sampled == 0:
                total_dis += abs(x.weight - ((v.weight - sum_weighted)/(v.num_examples - sampled)))
    return total_dis*0.5  # normalized as explained in the paper

def report_results(P):
    print("The pruning size is: " + str(np.size(P)))
    total_dis = 0
    max_dis = 0
    most_dis = P[0]
    for i in range(np.size(P)):
        print(P[i].id, end=" , ")
        print("discrepancy : " + str(P[i].Tdiscrepancy) + " , Number of queries: " + str(P[i].num_sampled))
        total_dis = total_dis + P[i].Tdiscrepancy
        if P[i].Tdiscrepancy > max_dis:
            max_dis = P[i].Tdiscrepancy
            most_dis = P[i]
    max_dis = 0
    ap_most_dis = P[0]
    for i in range(np.size(P)):
        # print(P[i].id, end = " , ")
        if P[i].approx_discrepancy > max_dis:
            max_dis = P[i].approx_discrepancy
            ap_most_dis = P[i]
    print("Most discrepancy by our estimate: " + ap_most_dis.id + " which is " + str(ap_most_dis.id == most_dis.id))

    print("The discrepancy of pruning achieved is: " + str(total_dis))
    print("***********************************************************")


# c - cost of high order weight query of a node. default is 0.
def AWP(root, Pruning, c):
    # splitting root node
    P = list(root.children)
    row = ['AWP']
    budget = 0 # counts the number of weight queries used
    for k in range(np.size(Pruning)): # return the state of algorithm at designated stopping points
        while np.size(P) < Pruning[k]:
            # Selecting a node to be sampled
            # n_to_query = select_nodes_LUCB(P) - different version
            n_to_query = select_nodes_UCB(P)
            for i in range(np.size(n_to_query)):
                budget = budget + query_node(n_to_query[i])
                update_approxdis_and_delta_Berstein(n_to_query[i], Pruning[k])
            # Selecting a node to be split (if one exist)
            l_t = select_nodes_UCB(P)[0]
            val = l_t.approx_discrepancy + l_t.delta
            flag = True
            options = []
            for i in range(np.size(P)):
                if Beta * (P[i].approx_discrepancy - P[i].delta) > val:
                    options = options + [i]
                elif Beta * (l_t.approx_discrepancy - l_t.delta) < P[i].approx_discrepancy + P[i].delta:
                    flag = False
            # chose one among the candidates for split
            if len(options) > 0:
                i = random.choice(options)
                if P[i].is_leaf:
                    print("problem - leaf selected to be split")
                else:
                    children = list(P.pop(i).children)
                    P = P + children
                    budget = budget + c
            elif flag: # split l_t in the appropriate case
                children = list(l_t.children)
                P.remove(l_t)
                P = P + children
                budget = budget + c
        row = row + [calculate_dis_modified(P), budget]
        print("Arrived to pruning of size " +str(len(P)))
        sys.stdout.flush()
        sys.stderr.flush()
    return row

# resets the tree before additional experiments
def reboot_tree(root):
    root.num_sampled = 0
    root.approx_discrepancy = 0
    root.delta = 2

    mid_nodes = list(root.children)
    while len(mid_nodes) > 0:
        if not mid_nodes[0].is_leaf:
            mid_nodes = mid_nodes + list(mid_nodes[0].children)
            mid_nodes[0].num_sampled = 0
            mid_nodes[0].messeured = 0
            mid_nodes[0].messeuredBy = 0
            mid_nodes[0].approx_discrepancy = 0
            mid_nodes[0].delta = np.inf
        else:
            mid_nodes[0].num_sampled = 0
        mid_nodes.pop(0)


def base_line_EMPIRICAL(root, b, s): # b is the budget for example weight queries, s is the final pruning size required
    P = [root]
    leafs = get_leaves(root)
    # querying
    for k in range(b):
        x = random.choice(leafs)
        val = x.weight
        leafs.remove(x)
        x.num_sampled += 1
        while x.id != root.id:
            x = x.parent
            x.num_sampled += 1
            if x.num_examples == x.num_sampled:
                x.approx_discrepancy = x.Tdiscrepancy
                x.delta = 0
            else:
                x.messeured += abs(val - x.weight / x.num_examples)
                x.approx_discrepancy = (x.num_examples / x.num_sampled) * x.messeured
    # split iteratively the most discrepancy by the naive estimator approximation
    for i in range(s - 1):
        most_dis = 0
        val = P[0].approx_discrepancy
        for j in range(np.size(P)):
            if P[j].approx_discrepancy > val:
                most_dis = j
                val = P[j].approx_discrepancy
        children = list(P[most_dis].children)
        P.pop(most_dis)
        P = P + children
    return [ calculate_dis_modified(P)]



def base_line_UNIFORM(root, b, s): # b is the budget for example weight queries, s is the final pruning size required
    P = [root]
    leafs = get_leaves(root)
    # querying
    for k in range(b):
        x = random.choice(leafs)
        val = x.weight
        leafs.remove(x)
        x.num_sampled += 1
        while x.id != root.id:
            x = x.parent
            x.num_sampled += 1
            if x.num_examples == x.num_sampled:
                x.approx_discrepancy = x.Tdiscrepancy
                x.delta = 0
            else:
                x.messeured += abs(val - x.weight / x.num_examples) - val
                x.approx_discrepancy = x.weight + (x.num_examples / x.num_sampled) * x.messeured
    # split iteratively the most discrepancy by our approximation
    for i in range(s - 1):
        most_dis = 0
        val = P[0].approx_discrepancy
        for j in range(np.size(P)):
            if P[j].approx_discrepancy > val:
                most_dis = j
                val = P[j].approx_discrepancy
        children = list(P[most_dis].children)
        P.pop(most_dis)
        P = P + children
    return [ calculate_dis_modified(P)]


def base_line_WEIGHT(root,b, s): # b is the budget for example weight queries, s is the final pruning size required
    P = [root]
    leafs = get_leaves(root)
    # querying
    for k in range(b):
        x = random.choice(leafs)
        val = x.weight
        leafs.remove(x)
        x.num_sampled += 1
        while x.id != root.id:
            x = x.parent
            x.num_sampled += 1
    # split iteratively the most weighted
    for i in range(s - 1):
        most_weighted = 0
        val = P[0].approx_discrepancy
        for j in range(np.size(P)):
            if P[j].weight > val:
                most_weighted = j
                val = P[j].weight
        children = list(P[most_weighted].children)
        P.pop(most_weighted)
        P = P + children
    return [calculate_dis_modified(P)]

# not used eventually in experiments. An oracal that selects the node to split by the maximal improvement in discrepancy
def orcal_localimprovement(root,Pruning):
    # deep copy
    S = []
    for i in range(np.size(Pruning)):
        S = S + [Pruning[i]]
    P = [root]
    dis_list =["Oi"]
    while len(S) > 0:
        best_index = 0
        best_gain = 0
        for i in range(np.size(P)):
            if not P[i].is_leaf:
                gain = P[i].Tdiscrepancy - P[i].children[0].Tdiscrepancy - P[i].children[1].Tdiscrepancy
                if gain > best_gain:
                    best_index = i
                    best_gain = gain

        v = P.pop(best_index)
        P = P + list(v.children)
        if len(P) == S[0]:
            dis_list += [calculate_dis_modified(P),0]
            S.pop(0)

    return dis_list


# Main
print("Started")
sys.stdout.flush()
sys.stderr.flush()
Delta = 0.05
Beta = 4
Worse_vm = 0 # keep track of numerical fix in empirical bernstein calculations
input_string = list(sys.argv)
root = setup_test(input_string[1],input_string[2]) # the root of the input hierarchical tree
print("Experiment setup completed")
sys.stdout.flush()
sys.stderr.flush()

Pruning = [2]
Col = ["Algtype" , str(2) + "_discrepancy_obtained"  , str(2) + "_NumExamples"]
stepsize = int(input_string[4])
for i in range(int(int(input_string[5]) / stepsize)):
    Pruning = Pruning +[(i + 1)*stepsize ]
    Col = Col + [ str((i + 1)*stepsize) + "_discrepancy_obtained"  , str((i + 1)*stepsize ) + "_NumExamples"]

df = pd.DataFrame(columns=Col) # will store the results

index = 0
for i in range(int(input_string[6])): # determine the number of repetitions for experiment
    print("Started the " + str(i+1) + "'th repetition")
    row = AWP(root, Pruning , 0)
    print("Done with AWP")
    sys.stdout.flush()
    sys.stderr.flush()
    df.loc[index] = row
    index += 1
    base_row = ['EMPIRICAL']
    smart_base_row = ['UNIFORM']
    base_weighted = ['WEIGHT']
    # base_Ua_row = ['U_a']
    for t in range(np.size(Pruning)): #running the baselines with the same number of example weight queries as used by AWP for each K size
        reboot_tree(root)
        base_row = base_row + base_line_EMPIRICAL(root,  int(row[(t+1)*2]), Pruning[t])+ [ int(row[(t+1)*2])]
        reboot_tree(root)
        smart_base_row = smart_base_row + base_line_UNIFORM(root, int(row[(t + 1) * 2]), Pruning[t])+ [ int(row[(t + 1) * 2])]
        reboot_tree(root)
        base_weighted = base_weighted + base_line_WEIGHT(root, int(row[(t + 1) * 2]), Pruning[t]) + [int(row[(t + 1) * 2])]

    df.loc[index] = base_row
    df.loc[index+1] = smart_base_row
    df.loc[index+2] = base_weighted
    index += 3
    reboot_tree(root)
    print("Done with repetition number " + str(i))
df.to_csv(input_string[3] )
