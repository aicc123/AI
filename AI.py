graph = {
    'A': ['B', 'C', "D"],
    'B': ['E', "F"],
    'C': ['G', "I"],
    'D': ["I"],
    'E': [],
    "F": [],
    'G': [],
    "I": []
}
 
def bfs(visit_complete, graph, current_node):
    visit_complete.append(current_node)
    queue = []
    queue.append(current_node)
 
    while queue:
        s = queue.pop(0)
        print(s)
 
        for neighbour in graph[s]:
            if neighbour not in visit_complete:
                visit_complete.append(neighbour)
                queue.append(neighbour)
 
bfs([], graph, 'A')





graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited = set() # Set to keep track of visited nodes.

def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
dfs(visited, graph, 'A')





def aStarAlgo(start_node, stop_node):
         
        open_set = set(start_node) 
        closed_set = set()
        g = {} #store distance from starting node
        parents = {}# parents contains an adjacency map of all nodes
 
        #ditance of starting node from itself is zero
        g[start_node] = 0
        #start_node is root node i.e it has no parent nodes
        #so start_node is set to its own parent node
        parents[start_node] = start_node
         
         
        while len(open_set) > 0:
            n = None
 
            #node with lowest f() is found
            for v in open_set:
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
                    n = v
             
                     
            if n == stop_node or Graph_nodes[n] == None:
                pass
            else:
                for (m, weight) in get_neighbors(n):
                    #nodes 'm' not in first and last set are added to first
                    #n is set its parent
                    if m not in open_set and m not in closed_set:
                        open_set.add(m)
                        parents[m] = n
                        g[m] = g[n] + weight
                         
     
                    #for each node m,compare its distance from start i.e g(m) to the
                    #from start through n node
                    else:
                        if g[m] > g[n] + weight:
                            #update g(m)
                            g[m] = g[n] + weight
                            #change parent of m to n
                            parents[m] = n
                             
                            #if m in closed set,remove and add to open
                            if m in closed_set:
                                closed_set.remove(m)
                                open_set.add(m)
 
            if n == None:
                print('Path does not exist!')
                return None
 
            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                path = []
 
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
 
                path.append(start_node)
 
                path.reverse()
 
                print('Path found: {}'.format(path))
                return path
 
 
            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_set.remove(n)
            closed_set.add(n)
 
        print('Path does not exist!')
        return None
         
#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
    if v in Graph_nodes:
        return Graph_nodes[v]
    else:
        return None
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
def heuristic(n):
        H_dist = {
            'A': 11,
            'B': 6,
            'C': 99,
            'D': 1,
            'E': 7,
            'G': 0,
             
        }
 
        return H_dist[n]
 
#Describe your graph here  
Graph_nodes = {
    'A': [('B', 2), ('E', 3)],
    'B': [('C', 1),('G', 9)],
    'C': None,
    'E': [('D', 6)],
    'D': [('G', 1)],
     
}
aStarAlgo('A', 'G')





#                                        Kruskal's Algorithm
# Kruskal's algorithm in Python

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])

    # Search function

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    #  Applying Kruskal algorithm
    def kruskal_algo(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        for u, v, weight in result:
            print("%d - %d: %d" % (u, v, weight))


g = Graph(6)
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 4)
g.add_edge(1, 2, 2)
g.add_edge(1, 0, 4)
g.add_edge(2, 0, 4)
g.add_edge(2, 1, 2)
g.add_edge(2, 3, 3)
g.add_edge(2, 5, 2)
g.add_edge(2, 4, 4)
g.add_edge(3, 2, 3)
g.add_edge(3, 4, 3)
g.add_edge(4, 2, 4)
g.add_edge(4, 3, 3)
g.add_edge(5, 2, 2)
g.add_edge(5, 4, 3)
g.kruskal_algo()






# Python program to solve N Queen
# Problem using backtracking

global N
N = 4

def printSolution(board):
	for i in range(N):
		for j in range(N):
			print (board[i][j],end=' ')
		print()


# A utility function to check if a queen can
# be placed on board[row][col]. Note that this
# function is called when "col" queens are
# already placed in columns from 0 to col -1.
# So we need to check only left side for
# attacking queens
def isSafe(board, row, col):

	# Check this row on left side
	for i in range(col):
		if board[row][i] == 1:
			return False

	# Check upper diagonal on left side
	for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	# Check lower diagonal on left side
	for i, j in zip(range(row, N, 1), range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	return True

def solveNQUtil(board, col):
	# base case: If all queens are placed
	# then return true
	if col >= N:
		return True

	# Consider this column and try placing
	# this queen in all rows one by one
	for i in range(N):

		if isSafe(board, i, col):
			# Place this queen in board[i][col]
			board[i][col] = 1

			# recur to place rest of the queens
			if solveNQUtil(board, col + 1) == True:
				return True

			# If placing queen in board[i][col
			# doesn't lead to a solution, then
			# queen from board[i][col]
			board[i][col] = 0

	# if the queen can not be placed in any row in
	# this column col then return false
	return False

# This function solves the N Queen problem using
# Backtracking. It mainly uses solveNQUtil() to
# solve the problem. It returns false if queens
# cannot be placed, otherwise return true and
# placement of queens in the form of 1s.
# note that there may be more than one
# solutions, this function prints one of the
# feasible solutions.
def solveNQ():
	board = [ [0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0]
			]

	if solveNQUtil(board, 0) == False:
		print ("Solution does not exist")
		return False

	printSolution(board)
	return True

# driver program to test above function
solveNQ()






import re 
import long_responses as long 
def message_probability(user_message, recognised_words, 
single_response=False, required_words=[]): 
message_certainty = 0 
has_required_words = True 
# Counts how many words are present in each predefined message 
for word in user_message: 
if word in recognised_words: 
message_certainty += 1 
# Calculates the percent of recognised words in a user message 
percentage = float(message_certainty) / 
float(len(recognised_words)) 
# Checks that the required words are in the string 
for word in required_words: 
if word not in user_message: 
has_required_words = False 
break 
# Must either have the required words, or be a single response 
if has_required_words or single_response: 
return int(percentage * 100) 
else: 
CHAT BOT PROGRAME CODE
return 0 
def check_all_messages(message): 
highest_prob_list = {} 
# Simplifies response creation / adds it to the dict 
def response(bot_response, list_of_words, single_response=False, 
required_words=[]): 
nonlocal highest_prob_list 
highest_prob_list[bot_response] = 
message_probability(message, list_of_words, single_response, 
required_words) 
# Responses 
response('Hello Ajay ! \n Welcome to Microscan Brodband Services 
! \n How may I Help You?', ['hello', 'hi', 'hey', 'sup', 'heyo'], 
single_response=True) 
response('See you!', ['bye', 'goodbye'], single_response=True) 
response('I\'m doing fine, and you?', ['how', 'are', 'you', 
'doing'], required_words=['how']) 
response('You\'re welcome!', ['thank', 'thanks'], 
single_response=True) 
response('Best Plan', ['which', 'best', 'plan'], 
required_words=['which', 'what']) 
response('Thank you!', ['i', 'love', 'code', 'palace'], 
required_words=['code', 'palace']) 
# Longer responses 
response(long.R_ADVICE, ['give', 'advice'], 
required_words=['advice']) 
response(long.R_EATING, ['what', 'you', 'eat'], 
required_words=['you', 'eat']) 
response(long.R_PLANS, ['plans'], required_words=['plans']) 
best_match = max(highest_prob_list, key=highest_prob_list.get) 
# print(highest_prob_list) 
# print(f'Best match = {best_match} | Score: 
{highest_prob_list[best_match]}') 
return long.unknown() if highest_prob_list[best_match] < 1 else 
best_match 
# Used to get the response 
def get_response(user_input): 
split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower()) 
response = check_all_messages(split_message) 
return response 
# Testing the response system 
while True: 
print('Bot: ' + get_response(input('You: '))) 
Extension long_responses.py
import random 
R_EATING = "I don't like eating anything because I'm a bot 
obviously!" 
R_ADVICE = "If I were you, I would go to the internet and type 
exactly what you wrote there!" 
R_PLANS = "----------Microscan Broadband Plans---------\n\n 1. 
40Mbps Unlimited - 399per Year \n 2. 60Mbps, Unlimited - 699per 
Year \n 3. 100Mbps, Unlimited - 999per Year \n 4. Maintainance 
and Services \n " 
def unknown(): 
response = ["Could you please re-phrase that? ", 
"...", 
"Sounds about right.", 
"What does that mean?"][ 
random.randrange(4)] 
return response
