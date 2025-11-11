"""Matches participants based on time horizon probabilities and the distribution over room sizes."""
import Distributions as dst, numpy as np, itertools as it, pprint as pp, warnings, scipy, random, copy, uuid, math
from scipy.optimize import LinearConstraint
abcs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
warnings.simplefilter("ignore")

plr_id_type = int | str | uuid.UUID

def divisors(num: int) -> list[int]:
    idx, divisors_ = 1, []
    while idx <= num:
        if (num % idx==0):
            divisors_.append(idx)
        idx += 1
    return divisors_


def feasible(population_size: int, possible_group_sizes: list[int], print_: bool = False) -> bool:
    """
    Checks if there is a feasible solution for propose_room_sizes.  Use print_ to see how this works.
    """
    possible_group_sizes = sorted(list(set(possible_group_sizes)))
    ranges = [list(range(0, math.floor(population_size / size) + 1)) for size in possible_group_sizes]
    multiples = list(it.product(*ranges))

    for combo in multiples:
        print_str = ""
        summer = [possible_group_sizes[idx] * val for idx, val in enumerate(combo)]
        for jdx in range(len(possible_group_sizes)):
            print_str += f"{possible_group_sizes[jdx]} x {combo[jdx]} + "
        print_str = print_str[:-2] + f"= {sum(summer)}"
        if print_: print(print_str)

        if sum(summer) == population_size: 
            return True
        
    return False


def propose_room_sizes(split_rooms: list[list[str | uuid.UUID | int]], group_size_dist: list[int | float] = [0.0, 0.6, 0.2, 0.2]) -> list[int]:
    """
    Proposes room sizes according to a given distribution of group sizes. Each room accommodates a group of players.

    Ensures that the proposed room sizes can accommodate all players from the previous round.
    
    Arguments:
        • split_rooms: list[list[str | uuid.UUID | int]]; List of groups from previous round.
            Each group is a list of player identifiers which can be string, uuid.UUID or int.
        • group_size_dist: list[int | float]; Probability distribution of room sizes (default: [0.0, 0.6, 0.2, 0.2]).

    Returns:
        • list[int]: List of proposed room sizes.

    Raises:
        • ValueError: If the solution of infeasible because of the population size and group size distribution 
        • ValueError: If the provided split_rooms is not a list of list of user ids.
        • ValueError: If the provided group_size_dist is not a list of numbers.
    """
    try: 
        for old_group in split_rooms:
            for new_group in old_group:
                if not all(isinstance(player, (str, uuid.UUID, int)) for player in new_group):
                    raise ValueError("split_rooms must be a list of list of user ids.")
    except: raise ValueError("split_rooms must be a list of list of user ids.")

    if not all(isinstance(elem, (int, float)) for elem in group_size_dist):
        raise ValueError("group_size_dist must be a list of numbers.")
    
    group_size_dist = [round(elem / sum(group_size_dist), 4) for elem in group_size_dist]
    prev_group_sizes = sorted([len(new_group) for old_group in split_rooms for new_group in old_group], reverse=True)
    population_size = sum(prev_group_sizes)

    "Check feasibility of solution"
    if group_size_dist[0] == 0:
        if population_size % 2 == 1 and not [size for idx, size in enumerate(group_size_dist) if size > 0.0 and idx % 2 == 0]:
            raise ValueError(f"Infeasible solution: population size is odd ({population_size}) but all room sizes are even.")
        
        if all((idx > population_size for idx, size in enumerate(group_size_dist) if size > 0)):
            raise ValueError("Infeasible solution: all non-zero room size proportions are larger than the population size.")
        
        feasible_sizes = [idx for idx in range(1, len(group_size_dist) + 1) if group_size_dist[idx-1] > 0]
        population_factors = divisors(population_size)

        if any(group_size > population_size for group_size in feasible_sizes): 
            raise ValueError("Infeasible solution: Some group sizes exceed entire population size.")
            
        if not any((idx in population_factors for idx in feasible_sizes)):
            if not feasible(population_size=population_size, possible_group_sizes=feasible_sizes):
                raise ValueError("Infeasible solution: Population size cannot be divided into the feasible group sizes.")

    fits_everyone = False
    while not fits_everyone:
        group_sizes = []
        while sum(group_sizes) != population_size:
            group_sizes.append(random.choices(population=list(range(1, len(group_size_dist)+1)), weights=group_size_dist)[0])
            while sum(group_sizes) > population_size: group_sizes.pop(random.randrange(len(group_sizes)))
        sezis_pourg = sorted(group_sizes, reverse=True)

        if prev_group_sizes[0] < sezis_pourg[0]: 
            fits_everyone = True
        elif prev_group_sizes[0] > sezis_pourg[0]: 
            fits_everyone = False
        else:
            fits_everyone = True
            for idx in range(len(sezis_pourg)):
                if prev_group_sizes[idx] < sezis_pourg[idx]:
                    break
                elif prev_group_sizes[idx] > sezis_pourg[idx]:
                    fits_everyone = False
                    break
          
    return sorted(group_sizes)


class MultiKeyDict(dict):
    """
    A dictionary with multiple keys, including keys with letters (A-Z) or keys with numbers (1-26).

    For example, a key can be "1-2" or "AB" or (1, 2) or [1, 2]. 
    If the key is a string with numbers separated by "-", it will be converted to a tuple of integers.
    If the key is a string with letters (A-Z), it will be kept as is.
    If the key is a tuple or list, it will be kept as is.

    Methods:
        • _standardize_key(key): Converts key into a standard form.
        • __getitem__(key): Returns value associated with the key.
        • __setitem__(key, value): Sets value associated with the key.
    """
    def _standardize_key(self, key):
        if isinstance(key, str):
            if key.isalpha():
                key = "-".join(str(ord(c) - ord('A') + 1) for c in key)
            return tuple(map(int, key.split("-")))
        elif isinstance(key, (list, tuple)):
            return tuple(key)
        else: raise TypeError("Invalid key type. Key must be a string, list, or tuple.")
    
    def __getitem__(self, key):
        standardized_key = self._standardize_key(key)
        return super().__getitem__(standardized_key)
    
    def __setitem__(self, key, value):
        standardized_key = self._standardize_key(key)
        super().__setitem__(standardized_key, value)


def gradient_descent(function_, other_args=(), start=np.random.uniform(0, 1), learning_rate=0.01, 
    max_iter=1000, epsilon=1e-5, num_restarts=10, toler=1e-6, momentum=0.9, print_=True) -> float:
    """
    Implements gradient descent with momentum to find the minimum of a given function. 
    
    It uses numerical differentiation for gradient approximation.

    Arguments:
        • function_: function; A function whose minimum needs to be found.
        • other_args: tuple; Additional arguments to be passed to function_.
            Note: Put a comma after an argument to allow the unpacking operator to work correctly. 
        • start: float; The initial value for the search (default: randomly chosen ⊂ [0, 1]).
        • learning_rate: float; The learning rate for gradient descent (default: 0.01).
        • max_iter: int; The maximum number of iterations for the search (default: 1000).
        • epsilon: float; The step size for numerical differentiation (default: 1e-5).
        • num_restarts: int; The number of random restarts to perform (default: 5).
        • toler: float; The tolerance for convergence (default: 1e-6).
        • momentum: float; Escape local minima and avoid oscillations.

    Notes: For expensive functions, set epsilon and toler to 1e-4.

    Returns:
        • float: The x value at which the function attains its minimum value.
    """
    def approximate_gradient(funct, xval, epsilon=epsilon):
        return (funct(xval + epsilon, *other_args) - funct(xval - epsilon, *other_args)) / (2 * epsilon)

    best_x = None
    best_loss = float("inf")

    for restart in range(num_restarts):
        xval = start if restart == 0 else np.random.uniform(0, 1)
        velocity = 0

        for idx in range(max_iter):
            prev_x = xval
            current_learning_rate = learning_rate / (1 + 0.1 * idx)
            gradient = approximate_gradient(function_, xval)
            velocity = momentum * velocity + (1 - momentum) * gradient
            xval -= current_learning_rate * velocity
            xval = max(min(xval, 1), 0)  # Ensure xval remains within [0, 1]

            if abs(xval - prev_x) < toler: break

        loss = function_(xval, *other_args)
        if loss < best_loss:
            best_loss = loss
            best_x = xval

        if print_:
            str1 = f"Restart {restart + 1}: Minimum value of the function: "
            str2 = f"{round(loss, 6)} at x = {round(xval, 6)} with {idx} attempts."
            print(str1 + str2)

    if print_:
        print(f"Best result: Minimum value of the function: {round(best_loss, 6)} at x = {round(best_x, 6)}")

    return best_x


def determine_matches(self) -> list[list[bool]]:
    """
    Based on an adjacency matrix, this method determines whether a match occurs between each pair of participants.

    The method compares a random number to each entry in the adjacency matrix. 
    If the entry in the adjacency matrix is larger than the random number, 
    a match occurs between the corresponding pair of participants.
    
    Returns:
        • np.ndarray: A Boolean matrix of matches between pairs of participants.
    """
    rando = random.random()
    self.tfmat = np.full((self.nPlayers, self.nPlayers), False, dtype=bool)
    for idx in range(self.nPlayers):
        for jdx in range(self.nPlayers):
            if self.matrix[idx][jdx] > rando: 
                self.tfmat[idx][jdx] = True
    return self.tfmat


def thdict_and_pairs(matrix) -> tuple[MultiKeyDict, list, list]:
    """
    Constructs a dictionary of pairwise probabilities and a list of player pairs.

    It takes as input an adjacency matrix and constructs a dictionary and list based on the entries in the matrix.
    The dictionary maps each pair of players to their matching probability, while the list consists of all possible pairs of players.
    
    Returns:
        • thdict: MultiKeyDict; A dictionary mapping each pair of players to their matching probability.
        • pairs: list[tuple[int]]; A list of tuples, each tuple represents a pair of players.
        • pairsABCs: list[str]; A list of strings, each string represents a pair of players.
    """
    nPlayers = len(matrix) if isinstance(matrix, list) else int(np.sqrt(matrix.size))
    thdict, pairs, pairsABCs = MultiKeyDict(), [], []
    for idx in range(nPlayers):
        for jdx in range(nPlayers):
            if idx < jdx: 
                playerA, playerB = idx+1, jdx+1
                pairs.append((playerA, playerB))
                pairsABCs.append(f"{abcs[idx]}{abcs[jdx]}")
                thdict[f"{playerA}-{playerB}"] = matrix[idx][jdx]
                thdict[f"{abcs[idx]}{abcs[jdx]}"] = matrix[idx][jdx]
    return thdict, pairs, pairsABCs


def routes(pairsABCs) -> tuple[MultiKeyDict, MultiKeyDict]:
    """
    Returns all potential routes connecting each participant pair.

    For each pair of participants, this function finds all paths through the network that connect the two participants.
    This is critical for the operation of the matching function, which needs to be aware of all potential participant interactions.

    Arguments:
        • pairsABCs: list[str]; A list of pairs of participants, represented as strings.

    Returns:
        • routes: MultiKeyDict; A dictionary mapping each pair of participants to all routes connecting them.
        • route_steps: MultiKeyDict; A dictionary mapping each pair of participants to the steps needed for each route.

    Raises:
        • TypeError: If pairsABCs is not a list.
        • ValueError: If pairsABCs is an empty list.
    """
    if len(pairsABCs) == 0: return MultiKeyDict(), MultiKeyDict()
    nPlayers = max([max(letter) for letter in pairsABCs])
    nPlayers = abcs.index(nPlayers) + 1
    routes, route_steps = MultiKeyDict(), MultiKeyDict()
    for pair in pairsABCs:
        remaining = abcs[:nPlayers].replace(pair[0], "").replace(pair[1], "")
        combos = [''.join(letter) for idx in range(len(remaining)) for letter in it.combinations(remaining, idx+1)]
        routes[pair], route_steps[pair] = [pair] + [pair[0] + c + pair[1] for c in combos], []
        for route in routes[pair]:
            steps = [sorted(subroute) for subroute in zip(route, route[1:])]
            steps = [''.join(player) for player in steps]
            route_steps[pair].append(steps)
    return routes, route_steps


def key_to_alpha(key: str | list | tuple) -> str:
    """
    Converts the key into an alphabetical representation.

    This function is used to standardize the representation of keys, facilitating the manipulation and understanding of key-value pairs.
    It supports strings, tuples, and lists, which are converted into a representation based on alphabetical characters.

    Arguments:
        • key: Union[str, list, tuple]; The key to convert.

    Returns:
        • str: The alphabetical representation of the key.

    Raises:
        • TypeError: If the key is not a string, list, or tuple.
    """
    if isinstance(key, str) and key.isalpha():
        return key
    elif isinstance(key, (str, list, tuple)):
        if isinstance(key, str):
            key = tuple(map(int, key.split("-")))
        elif isinstance(key, list):
            key = tuple(key)
        return "".join(chr(n + ord('A') - 1) for n in key)
    else: raise TypeError("Invalid key type. Key must be a string, list, or tuple.")


def valid_matrix(matrix: any) -> bool:
    """
    Verifies that an adjacency matrix is properly formatted. Adjacency matrices are 2D 
    arrays of matching probabilities where each element is between 0 and 1 inclusive.

    Arguments:
        • matrix: any; The matrix to validate. Can be an instance of `AdjacencyMatrix`,
                      a string with specific keywords, a list of lists, or a numpy array.

    Returns:
        • bool: `True` if the matrix is valid, `False` otherwise.
        
    Raises:
        • ValueError: If the matrix format is not supported.

    """
    if isinstance(matrix, AdjacencyMatrix):
        matrix = matrix.matrix
    if isinstance(matrix, str) and matrix.split("-")[-1].isdigit() and \
            matrix.split("-")[0] in ["sample", "random", "ones"]:
        return True
    elif isinstance(matrix, list) and all(isinstance(row, list) and len(row) == len(matrix) \
        and all(isinstance(x, (float, int)) and (0 <= x <= 1) for x in row) for row in matrix):
        return True
    elif isinstance(matrix, np.ndarray) and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] \
        and issubclass(matrix.dtype.type, (np.integer, np.floating)) and (matrix >= 0).all() and (matrix <= 1).all():
        return True
    else:
        print(f"Invalid matrix detected: {matrix}")
        return False


class AdjacencyMatrix():
    """
    Represents a matrix of pairwise matching probabilities.

    This class encapsulates an adjacency matrix, which represents the probabilities of pairwise matches between participants.
    Each entry in the matrix is a floating point value representing the probability of a match between two participants.
    The matrix is symmetric because the probability of a match between participant A and participant B is the same as 
    the probability of a match between participant B and participant A.
    
    Methods:
        • round_: Rounds all probabilities in the adjacency matrix.
        • multiply_elements: Multiplies all elements of the adjacency matrix by a scalar.
        • __repr__/__str__: Provides a string representation of the adjacency matrix.
        • json_serializable_matrix: Reformats matrix so that it can be sent within a json file.
        • TFpermutations: Computes all possible match outcomes represented as a list of True/False values.
    """        
    def __init__(self, matrix, distribution_type="uniform", coefficients=[.5, .5]) -> None:
        if isinstance(matrix, AdjacencyMatrix): 
            matrix = matrix.matrix

        if not valid_matrix(matrix):
            error_message = "Invalid matrix format"
            if isinstance(matrix, str):
                error_message += ". Please provide a valid matrix or one of the string literals 'random-n', 'sample-n', 'ones-n'"
            else: error_message += ": "
            raise ValueError(error_message + ": ", matrix)
        
        self.matrix, self.distribution_type, self.coefficients = matrix, distribution_type, coefficients
        
        if isinstance(matrix, str): 
            nPlayers = int(matrix.split("-")[-1])

            if matrix.startswith("sample"):
                samples_needed, samples = nPlayers**2, [-1]
                while not all((0 <= x <= 1) for x in samples):
                    samples = list(dst.sample_distribution(distribution_type=self.distribution_type, 
                        interval=[0, 1], coefficients=self.coefficients, size=samples_needed))
                adjacency_matrix = np.ones(shape=(nPlayers, nPlayers), dtype=float)
                for idx in range(nPlayers):
                    for jdx in range(nPlayers):
                        adjacency_matrix[idx][jdx] = adjacency_matrix[jdx][idx] = samples.pop()
                self.matrix = matrix = adjacency_matrix      

            elif matrix.startswith("random"):
                def random_adjacency_matrix(n_players):
                    "Generates a random adjacency matrix to test with adjacency_matrix_room()."
                    adjacency_matrix = np.zeros(shape=(n_players, n_players))
                    for idx, jdx in it.combinations(range(n_players), 2):
                        adjacency_matrix[idx][jdx] = adjacency_matrix[jdx][idx] = np.random.uniform()
                    return adjacency_matrix
                
                self.matrix = matrix = random_adjacency_matrix(n_players=nPlayers)

            elif matrix.startswith("ones"):
                self.matrix = matrix = np.ones(shape=(nPlayers, nPlayers), dtype=float)

        nPlayers = len(matrix)
        matrix_dimensions = np.shape(matrix)

        "Creating matrix from list of probabilities."
        if len(matrix_dimensions) == 1:
            self.matrix = [round(random.random(), 4) \
                if prob is None else prob for prob in self.matrix]
            nPlayers, nProbs = 0, 0
            while nProbs < matrix_dimensions[0]:
                nPlayers += 1
                nProbs_ = nPlayers**2 - nPlayers
                if nProbs_ > matrix_dimensions[0]: break
                nProbs = nProbs_
            self.matrix = self.matrix[:nProbs]
            matrix_ = np.ones(shape=(nPlayers, nPlayers), dtype=float)
            combos = list(it.combinations(iterable=list(range(nPlayers)), r=2))
            for coord, prob in zip(combos, matrix):
                matrix_[coord[0], coord[1]] = prob
                matrix_[coord[1], coord[0]] = prob
            self.matrix = matrix_

        "Ensuring that the matrix is symmetrical"
        if len(matrix_dimensions) == 2:
            nPlayers = len(matrix[0])
            if len(matrix) != nPlayers:
                raise ValueError("matrix must be square.")
            matrix = np.asarray(matrix)
            for idx in range(nPlayers):
                for jdx in range(nPlayers):
                    if idx == jdx: matrix[idx][jdx] = 1.0
                    elif idx > jdx: matrix[idx][jdx] = matrix[jdx][idx] = \
                        max(matrix[idx][jdx], matrix[jdx][idx])
            self.matrix = matrix

        self.nPlayers = nPlayers

        "Dictionary of pairwise probabilities and list of player pairs"
        self.thdict, self.pairs, self.pairsABCs = thdict_and_pairs(matrix=self.matrix)

        self.player_permutations = list(it.permutations(list(range(1, nPlayers + 1)), r=2))

        "For each participant pair, all the routes through the network connecting them."
        self.routes, self.route_steps = routes(pairsABCs=self.pairsABCs)


    def round_(self, ndigs: int = 4) -> None:
        """
        Rounds all probabilities in the adjacency matrix to a specified number of decimal places.

        Arguments:
            • ndigs: int; The number of decimal places (default: 4).
        """
        np.round(self.matrix, decimals=ndigs)


    def multiply_elements(self, scalar: float) -> None:
        """
        Multiplies all elements in the adjacency matrix by a given scalar.

        This method updates the adjacency matrix in-place by multiplying all elements by a scalar value. 
        It also updates the relevant attributes of the class instance to reflect the modified matrix.

        Arguments:
            • scalar: float; The scalar to multiply the matrix elements by.
        """
        for idx in range(self.nPlayers):
            for jdx in range(self.nPlayers):
                if idx > jdx: 
                    self.matrix[idx][jdx] = self.matrix[jdx][idx] = round(self.matrix[idx][jdx] * scalar, 4)

        "Updating attributes"
        self.thdict, self.pairs, self.pairsABCs = thdict_and_pairs(matrix=self.matrix)
        self.routes, self.route_steps = routes(pairsABCs=self.pairsABCs)


    def __repr__(self) -> str:
        """
        Returns a json serializable representation of the adjacency matrix.

        Returns:
            • list[list[float]]: A list of list of matching probabilities.
        """
        return repr(self.json_serializable_matrix())


    def __len__(self) -> int:
        """
        Returns the length of the matrix, equal to the number of players.
        """
        return len(self.matrix)


    def json_serializable_matrix(self) -> list[list[float]]:
        """
        Returns a json serializable representation of the adjacency matrix.

        Returns:
            • list[list[float]]: A list of list of matching probabilities.
        """
        json_serializable_matrix = []
        for idx in range(self.nPlayers):
            json_serializable_array = []
            for jdx in range(self.nPlayers):
                json_serializable_array.append(round(self.matrix[idx][jdx], 4))
            json_serializable_matrix.append(json_serializable_array)

        return json_serializable_matrix


    def __iter__(self):
        """
        Returns just the matrix, without any attributes
        """
        return iter(self.matrix.tolist()) if isinstance(self.matrix, np.ndarray) else iter(self.matrix)


    def mprint(self) -> None: 
        """
        Prints the adjacency matrix to the console.
        """
        print(self)


    def TFpermutations(self) -> None:
        """
        Computes all possible match outcomes for the adjacency matrix.

        This method generates a list of all possible match outcomes, represented as lists of True/False values, 
        where True represents a match and False represents no match. It works only for matrices of size up to 6x6.

        Returns:
            • None

        Raises:
            • RuntimeError: If the matrix size is larger than 6x6.
        """
        if self.nPlayers < 6: 
            self.TFperms = list(it.product([True, False], repeat=len(self.pairs)))

            "A condensed form of TF_permutations that names which pairs were matched in each permutation."
            selection_scenarios = []
            for lst in self.TFperms:
                selection_scenario = []
                for jdx, prob in enumerate(lst):
                    if prob: selection_scenario.append(self.pairs[jdx])
                selection_scenarios.append(selection_scenario)

            self.matchPerms = selection_scenarios

        else: raise RuntimeError("TFpermutations will not work on matrices greater than 6 x 6 players.")


    def _forward_interdependencies(self) -> None:
        """
        Adjusts pairwise time horizon probabilities to account for interdependencies among players.

        This method is used to update the probabilities in the adjacency matrix to reflect the interconnectedness of 
        the players in a game room. It first calculates all possible match permutations and assigns a True or False 
        value to indicate whether a match occurred or not in each permutation. Subsequently, it adjusts the original 
        pairwise probabilities by accounting for scenarios where a certain player interacts with more than one player.
        It finally updates the adjacency matrix with these new probabilities. The resulting adjacency matrix accurately 
        represents the matching probabilities, considering all possible interactions among players.

        Returns:
            None: Modifies the adjacency matrix of the instance in-place.
        """
        AdjacencyMatrix.TFpermutations(self)
        "Matrix indicating who is matched, True, and not matched, False, in each permutation."
        selected_not_selected = np.full((len(self.TFperms), len(self.TFperms[0])), False, dtype=bool)
        for idx, lst in enumerate(self.TFperms):
            selection_scenario = self.matchPerms[idx]
            for jdx, prob in enumerate(lst):
                pair = self.pairs[jdx]
                route_sections = self.route_steps[pair]
                if pair in selection_scenario: selected_not_selected[idx][jdx] = True 
                elif len(selection_scenario) >= len(self.pairs) - 1: selected_not_selected[idx][jdx] = True 
                elif len(selection_scenario) <= 1: selected_not_selected[idx][jdx] = False
                else:
                    for step in route_sections:
                        scenarioABCs = [key_to_alpha(key=section) for section in sorted(selection_scenario)]
                        if sorted(step) == scenarioABCs: selected_not_selected[idx][jdx] = True
                        elif set(step).issubset(set(scenarioABCs)): selected_not_selected[idx][jdx] = True

        "Permutations of pairwise probabilities, meaning selected or not selected"
        prob_permutations = np.zeros(shape=(len(self.TFperms), len(self.TFperms[0])))
        for idx, lst in enumerate(self.TFperms):
            for jdx, prob in enumerate(lst):
                if self.TFperms[idx][jdx]: prob_permutations[idx][jdx] = self.thdict[self.pairs[jdx]]
                else: prob_permutations[idx][jdx] = 1 - self.thdict[self.pairsABCs[jdx]]

        "All the output pairwise probabilities"
        output_pair_probs = {pair: 0 for pair in self.pairsABCs}
        for idx, pair in enumerate(self.pairsABCs):
            output_probability = 1
            for jdx, prob_lst in enumerate(prob_permutations):
                if not selected_not_selected[jdx][idx]: output_probability -= np.prod(prob_lst)
            output_pair_probs[pair] = round(output_probability, 4)

        "Placing the output probabilities into the output matrix"
        output_matrix = np.zeros(shape=(self.nPlayers, self.nPlayers))
        for pair, prob in list(output_pair_probs.items()):
            output_matrix[abcs.index(pair[0])][abcs.index(pair[1])] = output_matrix[\
                abcs.index(pair[1])][abcs.index(pair[0])] = round(prob, 4) if prob < 1 else 1.0
            
        np.fill_diagonal(output_matrix, 1.0)
        self.matrix = output_matrix

        "Updating attributes"
        self.thdict, self.pairs, self.pairsABCs = thdict_and_pairs(matrix=self.matrix)
        self.routes, self.route_steps = routes(pairsABCs=self.pairsABCs)
    

    def _reverse_interdependencies(self) -> None:
        """
        Reverses the process of accounting for interdependencies in the adjacency matrix.

        This method is used to approximate the initial pairwise probabilities before accounting for interdependencies. 
        It does this by using gradient descent to iteratively optimize an approximation of the original matrix. At each step, 
        it calculates a loss function, which is the root mean squared error (RMSE) between the current matrix and the target 
        matrix. It continues to adjust the matrix until the loss is minimized, indicating that the current matrix is a close 
        approximation of the original matrix.

        Returns:
            None: Modifies the adjacency matrix of the instance in-place.
        """
        def approx_matrix(xval, previous_matrix):
            """
            Because we currently lack a perfect algorithm for reversing the probability interdependencies,
            we must approximate this process by trial and error using gradient descent via the following steps:  
            (1) Copy the previous matrix and multiply all elements by a random x value between 0 and 1. (2) Update
            the probabilities of this copied matrix using copy_matrix._forward_interdependencies(). (3) Calculate
            loss by the average squared difference between all elements of the same index between both matrices.
            (4) Repeat process using gradient descent to approximate the x value resulting in the minumum loss.
            """ 
            copy_matrix = copy.deepcopy(previous_matrix)
            copy_matrix = AdjacencyMatrix(matrix=copy_matrix)
            copy_matrix.multiply_elements(scalar=xval)
            copy_matrix._forward_interdependencies()

            total_elements, sum_of_squared_residuals = 0, 0
            for idx in range(self.nPlayers):
                for jdx in range(self.nPlayers):
                    sum_of_squared_residuals += (previous_matrix[idx][jdx] - copy_matrix.matrix[idx][jdx])**2  
                    total_elements += 1  
            loss = np.sqrt(sum_of_squared_residuals / total_elements)   
            return loss    

        best_x = gradient_descent(function_=approx_matrix, other_args=(self.matrix,), start=np.random.uniform(0, 1), learning_rate=0.01, 
            num_restarts=int(30/self.nPlayers), max_iter=int(60/self.nPlayers), epsilon=1e-5, toler=1e-6, momentum=0.55, print_=False)
        AdjacencyMatrix.multiply_elements(self, scalar=best_x)


    def shift_probabilities(self, action="forward"):
        """
        Adjusts the adjacency matrix to either account for or ignore interdependencies.

        This method applies or reverses the process of accounting for interdependencies in the adjacency matrix, 
        depending on the provided action argument. It calls the _forward_interdependencies() method if the action is 
        "forward", or the _reverse_interdependencies() method if the action is "reverse".

        Arguments:
            • action: str; The action to perform ("forward" or "reverse").

        Returns:
            None: Modifies the adjacency matrix of the instance in-place.

        Raises:
            • ValueError: If the provided action is not "forward" or "reverse".
            • RuntimeError: If the number of players is greater than 6.
        """
        if self.nPlayers < 7:
            if action == "forward": AdjacencyMatrix._forward_interdependencies(self)
            elif action == "reverse": AdjacencyMatrix._reverse_interdependencies(self)
            else: raise ValueError("Invalid action. Use 'apply' or 'reverse'.")
        else: print(f"AdjacencyMatrix.shift_probabilities() is too slow to work on groups of {self.nPlayers} x {self.nPlayers}!")
        

    def determine_matches(self):
        """
        Updates the adjacency matrix to represent definitive matches.

        This method transforms the adjacency matrix to represent definite matches between players, replacing probabilities 
        with True (indicating a match) or False (indicating no match). The results are stored in the attribute 'tfmat'.

        Returns:
            • np.ndarray: A Boolean matrix representing definite matches between players.
        """
        self.tfmat = determine_matches(self)
        return self.tfmat


    def __getattr__(self, name):
        """
        Customizes attribute access for the AdjacencyMatrix class.

        This method overrides the default attribute access behavior for the class, providing custom functionality for 
        certain attributes. Specifically, it allows for 'tfmat' to always return the latest match results, 'matchPerms' 
        and 'TFperms' to return permutations of match results, and 'players' to verify the correct number of players.

        Arguments:
            • name: str; The name of the attribute being accessed.

        Returns:
            • Various: The value of the requested attribute.

        Raises:
            • AttributeError: If the requested attribute is not one of the customized attributes, or if 'players' does not have the correct number of players.
        """
        if name == 'tfmat':
            "tfmat will always return the same True-False values unless you reset them with self.determine_matches()."
            self.determine_matches()
            return self.tfmat
        elif name == 'matchPerms':
            self.TFpermutations()
            return self.matchPerms
        elif name == 'TFperms':
            self.TFpermutations()
            return self.TFperms
        else: raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def assign_to_game_rooms(split_rooms: list[list[plr_id_type]], group_sizes: list[int]) -> list[list[plr_id_type]]:
    """
    Assigns players to game rooms in a new round, subject to specific matching constraints.

    This function organizes players into game rooms, ensuring that those who must be matched together
    in the next round (according to 'split_rooms') are kept together, and those who must not be matched
    are separated. The distribution of players across rooms is determined by 'group_sizes'. The player 
    assignment process is conducted randomly, but always respects the matching constraints.

    The function first creates a binary 'A' matrix to represent the potential room assignments for each
    player subgroup, and then uses mixed-integer linear programming (MILP) to solve for the optimal room 
    assignments that satisfy the constraints. 

    Arguments:
        • split_rooms: list[list[plr_id_type]]; A nested list, each inner list contains player IDs which
            must be or must not be matched together in the next round.
        • group_sizes: list[int]; A list indicating the required number of players in each game room.

    Returns:
        • list[list[plr_id_type]]: A nested list, each inner list contains the player IDs assigned to a particular game room.

    Raises:
        ValueError: If any of the inputs are incorrectly formatted or incompatible.
    """
    subgroups = []
    prevgroupings = []
    counter = 0
    for idx in split_rooms:
        prevgroupings.append([])
        for jdx in idx:
            subgroups.append(len(jdx))
            prevgroupings[-1].append(counter)
            counter+=1
    A = []
    for idx in range(len(subgroups)):
        A.append([])
        for jdx in range(len(group_sizes)):
            for kdx in range(len(subgroups)):
                A[-1].append(int(idx==kdx))
    for idx in range(len(prevgroupings)):
        for jdx in range(len(group_sizes)):
            A.append([])
            for kdx in range(len(group_sizes)):
                B = [0]*len(subgroups)
                if (jdx==kdx):
                    for gdx in prevgroupings[idx]:
                        B[gdx]=1
                A[-1] = A[-1] + B
    for idx in range(len(group_sizes)):
        A.append([])
        for jdx in range(len(group_sizes)):
            for kdx in range(len(subgroups)):
                A[-1].append((idx==jdx)*subgroups[kdx])
    
    b_u = np.array(([1]*(len(subgroups)+len(group_sizes)*len(prevgroupings)))+group_sizes, dtype=np.float64)
    # b_u = np.array(([1]*(len(subgroups)+len(group_sizes)*len(prevgroupings)))+group_sizes)
    b_l = np.full_like(b_u, -np.inf)
    constraints = LinearConstraint(A, b_l, b_u)
    
    x = scipy.optimize.milp(c=-np.ones(len(subgroups)*len(group_sizes)),
                        integrality=np.ones(len(subgroups)*len(group_sizes)),
                        constraints=constraints).x
    
    new_groupings = []
    for idx in range(len(group_sizes)):
        new_groupings.append([])
        for jdx in range(len(subgroups)):
            if x[idx*len(subgroups)+jdx]:
                new_groupings[-1].append(jdx)
                
    flattened = []
    for idx in split_rooms:
        for jdx in idx:
            flattened.append(jdx)
                
    new_groupings2 = []
    for idx in new_groupings:
        new_groupings2.append([])
        for jdx in idx:
            new_groupings2[-1]+=flattened[jdx]
    return new_groupings2


def grouping_satisfies_constraint(new_room_assignment: list[list[plr_id_type]], split_rooms: list[list[plr_id_type]], group_sizes: list[int]) -> bool:
    """
    Checks whether the output of assign_to_game_rooms satisfies all constraints.

    This function validates the proposed room assignments from 'assign_to_game_rooms' against
    the constraints defined by 'split_rooms' and 'group_sizes'. This includes checks for unique
    player assignments (no player is assigned to more than one room), and the correct room sizes.

    Arguments:
        • new_room_assignment: list[list[int]]; The proposed assignment of players to rooms.
        • split_rooms: list[list[int]]; A nested list indicating which players must be or must not be matched.
        • group_sizes: list[int]; A list indicating the required number of players in each game room.

    Returns:
        • bool: True if the proposed room assignment satisfies all constraints, False otherwise.
    """

    if new_room_assignment is None:
        new_room_assignment = assign_to_game_rooms(split_rooms=split_rooms, group_sizes=group_sizes)

    new_room_assignment = sorted(new_room_assignment, key=len)
    group_sizes = sorted(group_sizes)

    "Checking if all players are unique"
    players_list, players_set = [], set()
    for new_room in new_room_assignment:
        for player in new_room:
            players_list.append(player)
            players_set.add(player)
    if len(players_list) != len(players_set):
        return False
    
    "Checking if each room is of the desired size"
    for room_size, new_room in zip(group_sizes, new_room_assignment):
        if len(new_room) != room_size: return False

    return True


def handle_degenerate_case(leftover_matches: list[list[list[plr_id_type]]], group_size_dist: list[float]) -> list[list[plr_id_type]]:
    """
    Proposes new room assignments when 'assign_to_rooms' fails because the constraints are impossible to satisfy.  This occurs when, 
    for instance, there are only n players, the group_size_dist only permits n player groups, but these players have been split up.

    Arguments:
        • leftover_matches: list[list[list[plr_id_type]]]; A nested list indicating which players must be or must not be matched.
        • group_size_dist: list[float]; A list indicating the distribution of group sizes.

    Returns:
        • list[list[plr_id_type]]: A nested list of room assignments, or an empty list if no valid assignment is found.
    """
    non_zero_group_sizes = [(size + 1, proportion) for size, proportion in enumerate(group_size_dist) if proportion > 0]
    players_lst = [player for group in [subgroup for supergroup in leftover_matches for subgroup in supergroup] for player in group]
    return [players_lst] if len(non_zero_group_sizes) == 1 and len(players_lst) == non_zero_group_sizes[0][0] else []


def assign_to_rooms(leftover_matches: list[list[list[plr_id_type]]], group_size_dist: list[float]) -> list[list[plr_id_type]]:
    """
    Proposes new room assignments while satisfying match and group size constraints.

    This function repeatedly proposes new room assignments using 'assign_to_game_rooms' and
    validates them using 'grouping_satisfies_constraint' until it finds an assignment that 
    satisfies all constraints. If it fails to find a valid assignment within 'max_iter' 
    attempts, it returns an empty list.

    Arguments:
        • leftover_matches: list[list[list[plr_id_type]]]; A nested list indicating which players must be or must not be matched.
        • group_size_dist: list[float]; A list indicating the distribution of group sizes.

    Returns:
        • list[list[plr_id_type]]: A nested list of room assignments, or an empty list if no valid assignment is found.
    """
    constraints_satisfied = False
    counter, max_iter = 0, 20

    while not constraints_satisfied:
        group_sizes = propose_room_sizes(split_rooms=leftover_matches, group_size_dist=group_size_dist)
        new_room_assignment = assign_to_game_rooms(split_rooms=leftover_matches, group_sizes=group_sizes)
        constraints_satisfied = grouping_satisfies_constraint(new_room_assignment=new_room_assignment, 
                                                              split_rooms=leftover_matches, group_sizes=group_sizes)
        counter += 1

        if counter > max_iter: 
            return handle_degenerate_case(leftover_matches=leftover_matches, group_size_dist=group_size_dist)

    return new_room_assignment


def merge_common_elements(matches: list[list[plr_id_type]]) -> list[list[int]]:
    """
    Merges sublists with common elements into larger groups.

    This function takes a list of 'matches' and merges any sublists that share common elements,
    creating larger player groups. It is used to consolidate the output from the 'split_room_or_not' function,
    grouping together players who are matched with one another.

    Arguments:
        • matches: list[list[plr_id_type]]; A nested list, each sublist contains pairs or groups of player IDs that are matched.

    Returns:
        • list[list[plr_id_type]]: A nested list, each sublist contains a group of player IDs that share matches.
    """
    groups = []

    for match in matches:
        matched_indices = []

        for i, group in enumerate(groups):
            if set(match) & set(group):
                matched_indices.append(i)

        if matched_indices:
            merged_group = match
            for index in matched_indices:
                merged_group.extend(groups[index])
            merged_group = list(set(merged_group))

            for index in sorted(matched_indices, reverse=True):
                del groups[index]

            groups.append(sorted(merged_group))
        else: groups.append(match)

    return groups


def split_room_or_not(player_ids: list[plr_id_type], adjacency_matrix: \
        AdjacencyMatrix, probabilities_already_reversed=False) -> list[list[plr_id_type]]:
    """
    Determines which players should remain matched in the next round based on their time horizons.

    This function uses the 'adjacency_matrix' to determine the time horizons (probability of 
    future encounters) between pairs of players. If their time horizon is high, they are likely 
    to remain matched in the next round. The function returns a list of groups, where each group 
    contains players who will remain matched.

    If 'probabilities_already_reversed' is False, the function first 'reverses' the probabilities 
    in the adjacency matrix to account for their interdependencies.

    Arguments:
        • player_ids: list[int | str | uuid.UUID]]; A list of player IDs. Player IDs
            are unique identifiers that remain constant across rounds and rooms.
        • adjacency_matrix: AdjacencyMatrix; An object representing the time horizons between player pairs.
        • probabilities_already_reversed: bool; Indicates whether the probabilities in the adjacency matrix 
          have already been reversed to account for their interdependencies.

    Returns: 
        • list[list[plr_id_type]]: A nested list indicating which players will remain matched in the next round.

    Raises:
        ValueError: If the number of players in 'player_ids' doesn't match the number of players in 'adjacency_matrix'.
    """
    nPlayers = len(player_ids)
    if nPlayers != adjacency_matrix.nPlayers: 
        raise ValueError(f"N players in player ids {nPlayers} ≠ N players in matrix {adjacency_matrix.nPlayers}.")

    player_ids = sorted(player_ids)

    if not probabilities_already_reversed:
        adjacency_matrix.shift_probabilities("reverse")
    
    "Reset probabilities and determine matches."
    adjacency_matrix.determine_matches()
    matrix_to_bools = adjacency_matrix.tfmat
    
    matches = []
    for idx in range(nPlayers):
        for jdx in range(nPlayers):
            if idx < jdx:
                if matrix_to_bools[idx][jdx]:
                    pA_, pB_ = player_ids[idx], player_ids[jdx]
                    matches.append([pA_, pB_])

    loners = []
    for idx, player_id in enumerate(player_ids):
        n_matches = np.count_nonzero(matrix_to_bools[idx])
        if n_matches == 1: loners.append([player_id])

    merged_groups = merge_common_elements(matches)
    for loner in loners: merged_groups.append(loner)

    return merged_groups


def iter_split_rooms(adjacency_matrices: list[AdjacencyMatrix]):
    """
    Iterates 'split_room_or_not' over multiple adjacency matrices to determine next round matches.

    This function applies the 'split_room_or_not' function to each adjacency matrix in 'adjacency_matrices', 
    producing a list of matching groups for each game room in the next round.

    Arguments:
        • adjacency_matrices: list[AdjacencyMatrix]; A list of AdjacencyMatrix objects. Each matrix corresponds 
          to a game room and includes the player IDs for all players in that room.

    Returns: 
        • list[list[list[plr_id_type]]]: A nested list, where each sublist corresponds to a game room and contains groups 
          of players that will remain matched in the next round.
    """
    split_or_nots = []
    for matrix in adjacency_matrices:
        group = matrix.players
        splits = split_room_or_not(player_ids=group, 
            adjacency_matrix=matrix, probabilities_already_reversed=False)
        split_or_nots.append(splits)

    return split_or_nots


def apply_adjacency_matrices_to_rooms(rooms_lst:list[list[str | int]], group_size_dist: list[float | int], 
                                      distribution_type: str='uniform', coefficients: list[float, float]=[.5, .2], print_=False):
    """
    Assigns an adjacency matrix to each room indicating the time horizon probabilities between players.

    This function iterates over a list of groups (rooms) of players. For each group, it constructs an adjacency
    matrix that represents the time horizon probabilities between the players within the group. These time
    horizon probabilities are sampled from a specified distribution.

    Arguments:
        • rooms_lst: list[list[str | int]]; A list of groups of players, with each group representing a room.
        • distribution_type: str; The type of distribution from which to sample time horizon probabilities.
        • coefficients: list[float]; Coefficients for the time horizon probability distribution.
        • group_size_dist: list[float | int]; The desired distribution of room sizes.
        • print_: bool; If True, print the adjacency matrix for each room.

    Returns: 
        • matrices_per_room: list[AdjacencyMatrix]; A list of adjacency matrices representing time horizon
          probabilities within each room.
    """
    non_zero_group_sizes = [(size + 1, proportion) for size, proportion in enumerate(group_size_dist) if proportion > 0]
    degenerate_case = len(rooms_lst) == 1 and len(non_zero_group_sizes) == 1

    matrices_per_room = []
    for group in rooms_lst:
        roomSize = len(group)
        # adjacency_matrix = AdjacencyMatrix(matrix= f"sample-{roomSize}",
        #     distribution_type=distribution_type, coefficients=coefficients)
        adjacency_matrix = AdjacencyMatrix(matrix=f"ones-{roomSize}" if degenerate_case else f"sample-{roomSize}",
            distribution_type=distribution_type, coefficients=coefficients)
        adjacency_matrix.players = group
        matrices_per_room.append(adjacency_matrix)
        if degenerate_case: 
            print(f"Applied degenerate matrix", adjacency_matrix)

    if print_: 
        for matrix in matrices_per_room:
            matrix: AdjacencyMatrix
            matrix.mprint(), print("")

    return matrices_per_room


def matching(previous_matrices: list[AdjacencyMatrix], group_size_dist: list[float | int], time_horizon_dist: dict = {\
        "distribution_type": "uniform", "coefficients": [0.5, 0.5]}, n_rounds=1, print_=False) -> list[AdjacencyMatrix]:
    """
    Conducts matching of players into rooms over multiple rounds based on time horizon probabilities.

    This function iterates over multiple rounds of a game, at each round using the adjacency matrices 
    from the previous round to guide the splitting and formation of new player groups (rooms). Within each room, 
    players are matched based on a specified distribution of time horizon probabilities. The function returns 
    the matches and the adjacency matrices for each round.

    Arguments:
        • previous_matrices: list[AdjacencyMatrix]; Adjacency matrices from the  
            last round, each containing time horizon probabilities and player IDs.
        • group_size_dist: list[float | int]; The desired distribution of room sizes.
        • time_horizon_dist: dict; Defines the distribution of time horizon probabilities.
        • n_rounds: int; The number of rounds to conduct matching for.
        • print_: bool; If True, print the results for each round.

    Returns: 
        • matches_by_round: list[list[uuid.UUID]]; A list of lists, where each sublist represents a round 
          and contains groups of matched player IDs.
        • matrices_by_round: list[AdjacencyMatrix]; A list of lists, where each sublist represents a round
          and contains the adjacency matrices for that round.

    Raises:
        • ValueError: If 'previous_matrices' is not a list of AdjacencyMatrix instances.
        • ValueError: If 'group_size_dist' contains anything other that numbers ⊂ [0, 1].
    """
    if not isinstance(previous_matrices, list) or not all(isinstance(mat, AdjacencyMatrix) for mat in previous_matrices):
        raise ValueError("previous_matrices must be a list of AdjacencyMatrices.")

    for proportion in group_size_dist:
        if not isinstance(proportion, (int, float)): 
            raise ValueError(f"group_size_dist must contain all numbers, not {type(proportion)}.")
        if not (0 <= proportion <= 1): 
            raise ValueError(f"group_size_dist must contain only numbers ranging between 0 and 1, not {proportion}.")

    matches_by_round, matrices_by_round = [], []

    while n_rounds > 0:
        current_matches = []

        while not current_matches:
            "Recieves list of previous matches from the previous matrices and determines who to split up."
            split_rooms = iter_split_rooms(adjacency_matrices=previous_matrices)

            "Assigns players to rooms for this round."
            current_matches = assign_to_rooms(leftover_matches=split_rooms, group_size_dist=group_size_dist)    

        "Assigns new within-room time horizon probabilities."
        adjacency_matrices_by_room = apply_adjacency_matrices_to_rooms(rooms_lst=current_matches, 
            distribution_type=time_horizon_dist["distribution_type"], coefficients=time_horizon_dist["coefficients"], group_size_dist=group_size_dist) 
        
        "Appends the results to both lists and decriments the number of remaining iterations."
        previous_matrices = copy.deepcopy(adjacency_matrices_by_room)
        matrices_by_round.append(adjacency_matrices_by_room)       
        matches_by_round.append(current_matches)
        n_rounds -= 1

        if print_: 
            for match, matrix in zip(current_matches, adjacency_matrices_by_room):
                matrix.mprint(), print("")

    if print_:
        print(""), print("Matches by Round:")
        pp.pprint(matches_by_round)

    return matches_by_round, matrices_by_round


def initial_matches(user_ids: list[str | int]) -> list[AdjacencyMatrix]:
    """
    Prepares inputs for the first round of the matching function.

    This function initializes a list of AdjacencyMatrix instances, one for each player in the game. Each matrix is 
    initially a 1x1 matrix because the players have not yet been matched.

    Arguments:
        • user_ids: list[str | int]; A list of player IDs.

    Returns: 
        • round_zero_matrices: list[AdjacencyMatrix]; A list of adjacency matrices for the first round, one for each player.
    """
    round_zero_matrices = []
    for user_id in user_ids:
        adjacency_matrix = AdjacencyMatrix(matrix="random-1")
        adjacency_matrix.players = [user_id]
        round_zero_matrices.append(adjacency_matrix)

    return round_zero_matrices


def n_rounds(prob_slope=0.5, prob_midpoint=10, constant=5.5452) -> int:
    """
    Determines the number of rounds for the experiment based on a probabilistic calculation.

    This function implements a while-loop that increments the current round and determines the probability of moving 
    to the next round based on a defined function. The loop continues until the calculated probability is lower than 
    a random number or the maximum number of rounds is reached.

    Arguments:
        • prob_slope: float; The slope of the probability function for moving to the next round.
        • prob_midpoint: int; The midpoint of the probability function for moving to the next round.
        • constant: float; A constant value used in the probability function.

    Returns: 
        • int: The number of rounds for the experiment.

    Raises:
        Exception: If the maximum round limit (200) is reached.
    """
    current_round, max_rounds = 0, 200
    while max_rounds > 0:
        max_rounds -= 1
        current_round += 1
        p_next_round = dst.prob_next_round(current_round_number=current_round, 
            prob_slope=prob_slope, prob_midpoint=prob_midpoint, constant=constant)
        rando = random.random()
        if rando > p_next_round:
            return current_round

    raise Exception(f"Reset the slope and midpoint such that the experiment won't exceed 200 rounds.")


import plotly.graph_objects as go
def validate_matching_function(matches_by_round: list[list[int]], matrices_by_round: list[list[AdjacencyMatrix]], 
                               group_size_dist: list[float], time_horizon_dist: dict[str:str | list[float]]) -> go.Figure:
    """
    Tests the matching function by comparing expected to actual output. The distribution of matching probabilities should 
    align with the distribution specified by the researcher. This (mis)alignment can be seen visually by comparing the 
    desired distribution to a histogram of actual matching probabilities. Next, this checks if the matches in round n 
    correspond to the matching probabilities shown to the players in round n - 1. I have yet to figure this part out.

    Arguments:    
        • matches_by_round: list[list[uuid.UUID]]; A list of lists, where each 
          sublist represents a round and contains groups of matched player IDs.
        • matrices_by_round: list[AdjacencyMatrix]; A list of lists, where each 
          sublist represents a round and contains the adjacency matrices for that round. 
        • group_size_dist: list[float | int]; The desired distribution of room sizes.
        • time_horizon_dist: dict[str:str | list[float]]; Defines the distribution  
          of time horizon probabilities.

    Returns:
        • go.Figure: Figure comparing the two distributions.
    """
    "Extracting all matching probabilities."
    matching_probabilities = []
    for round_ in matrices_by_round:
        for amatrix in round_:
            for idx, row in enumerate(amatrix):
                for jdx, prob in enumerate(row):
                    if idx > jdx:
                        matching_probabilities.append(prob)

    # dst.plot_distribution(time_horizon_dist["distribution_type"], [0, 1], 
    #                       time_horizon_dist["coefficients"], matching_probabilities, True)
    print(""), print("Validate")
    prob_match_to_matches = []
    n_rounds = len(matches_by_round)
    for round_num in range(n_rounds-1):
        if round_num < n_rounds:
            round_match, round_matrix = matches_by_round[round_num], matrices_by_round[round_num]
            n_players_round = len(set([player for room in round_match for player in room]))
            relevance_matrix = [[False] * n_players_round for _ in range(n_players_round)] 
            match_amatrix = [[0] * n_players_round for _ in range(n_players_round)]
            for room_match, room_matrix in zip(round_match, round_matrix):
                for pair, comb in zip(room_matrix.pairs, list(it.combinations(room_match, 2))):
                    room_matrix = list(room_matrix)
                    match_probability = room_matrix[pair[0]-1][pair[1]-1]
                    match_probability = round(match_probability, 2)
                    match_amatrix[comb[0]-1][comb[1]-1] = match_probability
                    relevance_matrix[comb[0]-1][comb[1]-1] = True

            round_match_, round_matrix_ = matches_by_round[round_num+1], matrices_by_round[round_num+1]
            n_players_round_ = len(set([player for room in round_match_ for player in room]))
            match_amatrix_ = [[0] * n_players_round_ for _ in range(n_players_round_)]
            for room_match_ in round_match_:
                player_combos_ = list(it.combinations(room_match_, 2))
                for pcombo_ in player_combos_:
                    match_amatrix_[pcombo_[0]-1][pcombo_[1]-1] = 1

            for idx in range(len(match_amatrix)):
                row_amatrix, row_amatrix_ = match_amatrix[idx], match_amatrix_[idx]
                for jdx in range(len(row_amatrix)):
                    if relevance_matrix[idx][jdx]:
                        prob_match_to_matches.append((row_amatrix[jdx], row_amatrix_[jdx]))

            print("")
            for row in match_amatrix:
                print(row)
            print("")
            for row_ in match_amatrix_:
                print(row_)
            print("")
    print(prob_match_to_matches)
  
    return matching_probabilities


def latin_square(size: int) -> list[list[int]]:
    """Produces a symmetrical size x size latin square"""

    ls = np.zeros(shape=(size, size), dtype=object)

    def val(i, j, n=size):
        if (j == 0):
            return i
        if (i == n - 1):
            return ((n // 2) * (j - 1) % (n - 1))
        if (j == (2 * i) % (n - 1) + 1):
            return n - 1
        return (j - 1 - i) % (n - 1)

    for i in range(size):
        for j in range(size):
            ls[i][j] = val(i, j, size)

    return ls


def global_adjacency_matrix(size: int, time_horizon_dist: dict) -> list[list[float]]:
    """
    Generates and assigns probabilities to a latin square that governs matching 
    probilities across all rounds of an experiment.

    Arguments:
        • size: int; The number of players.
        • time_horizon_dist: dict; The shape of the probability distribution to sample.
            - Example: {
                'distribution_type': 'linear',
                'coefficients': [1.0, 0.5]
            }

    Returns: 
        • np.NDArray[float]: 2d array of pairwise matching probabilities.
    """
    "Type checking inputs."
    if not isinstance(size, int):
        raise TypeError(f"size({type(size)}) must be an integer >= 2.")
    if not size >= 2:
        raise ValueError(f"size({size}) must be greater than or equal to 2.")
    if not isinstance(time_horizon_dist, dict):
        raise TypeError(f"time_horizon_dist{type(time_horizon_dist)} must be a dictionary.")
    
    distribution_type = time_horizon_dist.get("distribution_type", None)
    if not isinstance(distribution_type, str):
        raise TypeError(f"distribution_type({type(distribution_type)}, {distribution_type}) must be a sting.")

    coefficients = time_horizon_dist.get("coefficients", None)
    if not isinstance(coefficients, list) or len(coefficients) != 2 or not isinstance(
        coefficients[0], (float, int)) or not isinstance(coefficients[1], (float, int)):
        raise ValueError(f"coefficients must be a list of two floats.")

    "Generating a list of matching probabilities based on the distribution type and coefficients."
    samples = list(dst.sample_distribution(distribution_type=distribution_type, 
        interval=[0, 1], coefficients=coefficients, size=size-1))

    "Normalizing probabilities so that they sum to 1."
    total = sum(samples)
    samples = [num / total for num in samples]      

    "Generating latin square of integers."
    lsquare = latin_square(size=size)

    "Extracting column of reference player numbers."
    reference_column = lsquare[:, 0]
    lsquare = lsquare[:, 1:]

    "Generating adjacency matrix of matching probabilities."
    lsquare_probs = np.zeros(shape=(size, size), dtype=object)
    for idx in range(size):
        lsquare_probs[idx][idx] = 1.0

    for idx in range(size):
        ref_player_num = reference_column[idx]
        ls_row = lsquare[idx]
        for jdx in range(size-1):
            match_prob = float(samples[jdx])
            pair = (ref_player_num, ls_row[jdx])
            lsquare_probs[pair[0]][pair[1]] = match_prob

    return lsquare_probs


def match_pairs(player_indices: list[int], adjacency_matrix: np.ndarray) -> list[tuple[int, int]]:
    """
    Matches players into pairs based on probabilities from an adjacency matrix.

    Arguments:
        • player_indices: List[int]; List of player indices (e.g., [0, 1, 2, ..., N-1]).
        • adjacency_matrix: np.ndarray; A square matrix of shape (N, N) where N is the number of players.
          Each element [i][j] represents the probability of player i being matched with player j.
          Diagonal elements should be disregarded (set to zero or any value, as players cannot be matched with themselves).

    Returns:
        • matches: List[Tuple[int, int]]; A list of tuples where each tuple represents a matched pair of player indices.
    
    Notes:
        • The function ensures that each player is matched with exactly one other player and no player is matched with themselves.
        • The matching is done for one round. Over multiple rounds, the frequencies of pairings should approximate the probabilities in the adjacency matrix.
        • The function uses a greedy approximation method due to computational constraints.

    Raises:
        • ValueError: If the number of players is not even.
    """
    num_players = len(player_indices)
    if num_players % 2 != 0:
        raise ValueError(f"The number of players ({num_players}) must be even.")

    "Initialize list to store matches"
    matches = []

    "Create a copy of the adjacency matrix to modify"
    prob_matrix = adjacency_matrix.copy()

    "Set diagonal to zero to prevent self-pairing"
    np.fill_diagonal(prob_matrix, 0)

    "Keep track of unmatched players"
    unmatched_players = set(player_indices)

    while unmatched_players:
        "Randomly select a player from the unmatched players"
        player = random.choice(list(unmatched_players))

        "Get probabilities for this player to be matched with other unmatched players"
        potential_partners = list(unmatched_players - {player})
        if not potential_partners:
            "No partners left (should not happen with even number of players)"
            break

        partner_probs = prob_matrix[player, potential_partners]
        "Normalize probabilities"
        total_prob = np.sum(partner_probs)
        if total_prob > 0:
            normalized_probs = partner_probs / total_prob
        else:
            "If total_prob is zero, assign equal probability"
            normalized_probs = np.full(len(potential_partners), 1 / len(potential_partners))
        normalized_probs = [float(prob) for prob in normalized_probs]

        "Randomly select a partner based on the probabilities"
        partner = np.random.choice(potential_partners, p=normalized_probs)

        "Add the match"
        matches.append((player, partner))

        "Remove both players from the unmatched set"
        unmatched_players.discard(player)
        unmatched_players.discard(partner)

    return matches


def matching_global(player_uuids: list[uuid.UUID], global_adjacency_matrix: list[list[float]], n_rounds=1, print_=False) -> list[AdjacencyMatrix]:
    """
    Matches player pairs into game rooms over multiple rounds based on matching probabilities. 

    Arguments:
        • player_uuids: list[uuid.UUID]; List of player identifiers.  
        • global_adjacency_matrix: list[list[float]]; Adjacency matrix of matching probabilies.
        • n_rounds: int; The number of rounds to conduct matching for.
        • print_: bool; If True, print the results for each round.

    Returns: 
        • matches_by_round: list[list[uuid.UUID]]; A list of lists, where each sublist represents a round 
          and contains groups of matched player IDs.
        • matrices_by_round: list[AdjacencyMatrix]; A list of lists, where each sublist represents a round
          and contains the adjacency matrices for that round.

    Notes:
        • While the other matching function matches n player groups based on probabilies that apply 
            one round at a time, this function matches 2 player groups based on probabilites that apply
            to all rounds of the experiment.

    Raises:
        • ValueError: If the number of players in 'player_uuids' is not an even number.
        • ValueError: If 'global_adjacency_matrix' dimensions do not match the number of players.
    """
    "Validating inputs."
    n_players = len(player_uuids)
    if n_players % 2 != 0:
        raise ValueError(f"The number of players({n_players}) must be even.")  

    num_rows, num_cols = global_adjacency_matrix.shape
    if not (num_rows == num_cols == n_players and np.array_equal(global_adjacency_matrix, global_adjacency_matrix.T)):
        raise ValueError(f"The number of players({n_players}) does not match the adjacency matrix shape[({num_rows}, {num_cols})].")

    "Array of tuples representing matched players and an array of AdjacencyMatrices."
    matches_by_round, matrices_by_round = [], []

    for round_num in range(n_rounds):
        "Generating player pairs for a given round."
        matches_this_round = match_pairs(player_indices=list(range(n_players)), 
                                           adjacency_matrix=global_adjacency_matrix)

        "Generates adjacency matrices, which are used to display matching probabilities on-screen."
        matrices_this_round = []
        for match in matches_this_round:
            player_row_idx, player_col_idx = match
            match_prob = global_adjacency_matrix[player_row_idx][player_col_idx]
            amatrix = AdjacencyMatrix([[1.0, match_prob], [match_prob, 1.0]])
            matrices_this_round.append(amatrix)

            if print_:
                amatrix.mprint()
                print("")

        matches_by_round.append(matches_this_round)
        matrices_by_round.append(matrices_this_round)

    if print_:
        print(""), print("Matches by Round:")
        pp.pprint(matches_by_round)

    return matches_by_round, matrices_by_round


# n_players = 4
# player_indices = list(range(n_players))
# time_horizon_dist = {"distribution_type": "linear", "coefficients": [1.0, 0.5]}
# amatrix = global_adjacency_matrix(size=n_players, time_horizon_dist=time_horizon_dist)
# print(match_pairs(player_indices=player_indices, adjacency_matrix=amatrix))
# matching_global(player_uuids=player_indices, global_adjacency_matrix=amatrix, n_rounds=66, print_=True)
# print(amatrix)


# "Example Use"
# population_size, use_uuids = 10, False
# group_size_dist_ = [0.0, 1.0, 0.0, 0.0, 0.0]
# time_horizon_dist_ = {"distribution_type": "linear", "coefficients": [1.0, 0.5]}
# n_rounds_ = n_rounds(prob_slope=1, prob_midpoint=10, constant=5.5452)
# round_zero_matrices = initial_matches(user_ids=[uuid.uuid4() if use_uuids else user for user in range(1, population_size+1)])
# matches_by_round, matrices_by_round = matching(previous_matrices=round_zero_matrices, group_size_dist=group_size_dist_, 
#     time_horizon_dist=time_horizon_dist_, n_rounds=n_rounds_, print_=True)
# matching_probabilities = validate_matching_function(matches_by_round, matrices_by_round, group_size_dist_, time_horizon_dist_)
# # print(matching_probabilities)


"Example Use"
# population_size, use_uuids = 8, False
# n_rounds_ = n_rounds(prob_slope=1, prob_midpoint=10, constant=5.5452)
# round_zero_matrices = initial_matches(user_ids=[uuid.uuid4() if use_uuids else user for user in range(1, population_size+1)])
# matches_by_round, matrices_by_round = matching(previous_matrices=round_zero_matrices, group_size_dist=[0.0, 1.0, 0.0, 0.0, 0.0], 
#     time_horizon_dist={"distribution_type": "uniform", "coefficients": [0.5, 0.5]}, n_rounds=n_rounds_, print_=True)