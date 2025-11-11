"""GameTree"""
# from . import Agent as ag
# from . import Match as match
# from . import Players as play
# import numpy as np, itertools as it, pandas as pd, \
#     pprint as pp, datetime, copy, random, json, os, re
import Match as match, Players as play, Agent as ag, numpy as np, itertools as it, pandas as \
    pd, plotly.graph_objects as go, pprint as pp, datetime, time, copy, random, json, os, re
abcs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + "".join([chr(code) for code in range(945,970)]) \
    + "".join([chr(code) for code in range(575,675)])
file_path_gametrees = "./Inputs/Trees/Json_Trees"
file_path_data = "./server/game_engine/Data"
from collections import Counter
from flask_socketio import emit

full_path = os.path.join("./", "verb_phrases.json")
if os.path.exists(full_path):
    with open(full_path, "r") as f:
        verb_phrases = json.load(f)

response_type = None | dict[str: dict[str: object]]
number = int | float

"""Data File Variables:"""
"Columns that are standard for every game tree and represent information that applies to the entire tree, rather than specific nodes."
general_columns = {
    'experiment_setting_keys': 'setting_keys', 'experiment_setting_values': 'setting_vals', 'round': 'round', 'room': 'room', 'batch': 'batch', 
    'timestamp': 'time', 'title': 'title', 'tree_tag': 'tag', 'player_uuids': 'uuids', 'player_types': 'player_types', 'avatar_shapes': 'avshapes', 
    'avatar_colors': 'avcolors', 'adjacency_matrix': 'matrix', 'idnum_label_map': 'id_map', 'perspective_of_player': 'perspective', 'final_node': 
    'final_node', 'tree_status': 'tree_status', 'players_abdicated': 'abdications', 'durations': 'durations', 'cumulative_payoffs_tree': 'cpayoffst', 
    'cumulative_payoffs_experiment': 'cpayoffse'
    }   

"Optional helper columns that are useful in standard types of data analysis"
analysis_columns = {
    'alignment': 'alignment', 'inequality': 'inequality', 'agency': 'agency', 'truthfulness': 
    'truthfulness', 'utilities': 'utilities', 'cooperation_rate': 'coop_rate'
    }

"If in expanded format, these are columns for each attribute of each node or even each item of each list attribute."
attribute_columns = {
    'choice_type': 'choice_type', 'probability': 'prob', 'positionxy': 'pos', 
    'info_set': 'info_set', 'time': 'time', 'beliefs': 'beliefs', 'options': 'options'
    }
list_attr_columns = {
    'payoffs': 'pay', 'choice': 'choo', 'prediction': 'pred', 'choice_data': 'choodata', 'prediction_data': 'preddata'
    }

text_color, text_family = "white", "Calibri"
fig_layout = {
    "template": "plotly_dark",
    "font": dict(family=text_family, color=text_color, size=24),
    "tickfont": dict(family=text_family, color=text_color, size=30),
    "titlefont_size": 48, "title_x": 0.5, "title_y": 0.96, "scale": ("x", 1),
    "colorscales": ['Viridis', 'Plasma', 'Inferno', 'matter', 'haline', 'thermal', 'dense', 'Magma'],
    "annotations": {"font":  dict(family=text_family, color=text_color, size=34), "showarrow": False}, 
    "xaxis" : {"title_font": dict(family=text_family, color=text_color, size=34), 
        "tickfont": dict(size=30, family=text_family, color=text_color)},
    "yaxis" : {"title_font": dict(family=text_family, color=text_color, size=34), 
        "tickfont": dict(size=30, family=text_family, color=text_color)},
    "hoverlabel": dict(font_size=30, font_family=text_family)
}

class MemoizationCache:
    """
    If a Node or Tree method is called repeatedly, they can store their 
    outputs in this memoization cache to avoid redundant computations.
    """

    def __init__(self) -> None:
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value


class Beliefs(dict):
    """
    A dictionary for storing and manipulating players' belief attributes for each node in a game tree.
    
    The beliefs are about various game attributes, such as payoffs, probabilities, choosers, etc., and 
    can also involve nested beliefs (beliefs about beliefs and so forth). Each node of the tree holds 
    a Beliefs instance, which tracks the deviations from the true state of the game tree as perceived 
    by the players. This means that players do not necessarily know the true state of the tree.

    Attributes
        • nodeid: int; The ID of the node that this Beliefs instance is attached to.
        • tree: Tree; The game tree that this Beliefs instance is part of.
        • supported_attributes: list;
            A list of attributes that are supported by the Beliefs class.

    Methods
        • __setitem__(self, key: str, value):
            Sets the belief attribute value for a given key.
        • __getitem__(self, key: str):
            Returns the belief attribute value for a given key.
    """   

    def __init__(self, nodeid: int, tree: 'Tree'):
        self.nodeid = nodeid
        self.tree = tree

        self.supported_attributes = [
            "players", 
            "payoffs", 
            "probability", 
            "chooser"
            ]

    def __setitem__(self, key: str, value):
        """
        Sets the belief attribute value for a given key. The method ensures the belief attribute 
        is stored at the correct node, and if the node isn't the current one, it redirects to the 
        correct node. If the key doesn't point to an attribute, it raises an error.
        
        Arguments
            • key: str; The key specifying the belief attribute to set.
            • value: The value to set for the belief attribute.

        Raises
            • ValueError: If the key doesn't contain an attribute.
        """

        start_node = self._starting_node(key)
        if start_node != self.nodeid:
            "Redirect to beliefs within the correct node."
            self.tree.nodes[start_node].beliefs[key] = value
        else: 
            key = self.formatted_key(key)
            if self._specificity_level(key) != "attribute":
                raise ValueError("key must contain the attribute.")

            super().__setitem__(key, value)

    def __getitem__(self, key: str) -> object:
        """
        Returns the belief attribute value for a given key.  If the attribute is not within
        the belief dictionary, then this method checks for the belief attribute recursively, 
        deferring to the parent node if the attribute isn't found at the current node.  This 
        deferment continutes until the root node.  If the attribute isn't found at the root
        node, the 'belief level is reduced' by reduce_belief_level.  This process continues 
        until the belief is at the lowest order.  Finally, if the attribute is not stored 
        in any belief dictionary, this method returns the objective attribute.  

        Levels of Beliefs
        Beliefs exist at different levels.  
        For example, the belief order of 'I believe that you believe that I believe that...' 
        is higher than the belief order of 'I believe that...'  Furthermore, a beliefs can 
        pertain to nodes past, present, or future.  The key notation indicates the belief 
        order and location.  For example: '@n6~p1b:@n3~p2b:...' means that 'At node 6, 
        player 1 believes that at node 3, player 2 believed that...'

        Arguments
            • key: str; The key specifying the belief attribute to get.

        Returns: 
            • object: The value of the belief attribute specified by the key.

        Raises:
            • ValueError: If the key doesn't contain an attribute.
               
        """

        key = self.formatted_key(key)
        
        if self._specificity_level(key) != "attribute":
            "TODO consider finding a way to return entire node."
            raise ValueError("key must contain the desired attribute.")       

        attribute = self._attrinkey(key)
        
        if "~" in key and attribute not in self.supported_attributes:
            "If attribute is not supported, return the objective attribute."
            return getattr(self.tree.nodes[self.nodeid], attribute)

        if self.tree.nodes is None:
            if key in self: 
                return super().__getitem__(key)
            return getattr(self, attribute)

        while True:
            """
            If the believed attribute is saved, return it.  Otherwise, defer to the parent.  If at 
            root, defer to a lower-order belief.  At the base case, defer to the objective attribute.
            """
            start_node = self._starting_node(key)
            if start_node != self.nodeid:
                "Redirect to beliefs within the correct node."
                return self.tree.nodes[start_node].beliefs[key]

            if self._base_case_key(key):
                attribute = self._attrinkey(key)
                current_node = int(key.split("~")[0][2:])
                this_node = self.tree.nodes[current_node]

                "Temporarily resetting perspective to objective to prevent infinite recursion."
                perspective = copy.copy(self.tree.perspective_of_player)
                self.tree.perspective_of_player = None
                attribute = getattr(this_node, attribute)
                self.tree.perspective_of_player = perspective

                return attribute
            
            if key in self: 
                return super().__getitem__(key)
            
            key = self.reduce_belief_level(key)     

    def _starting_node(self, key: str | list[int | str]) -> int:
        """
        Returns the first nodeid within the key.
        
        Arguments
            • key: str | list[int | str]; Key for accessing beliefs in the
                belief dictionary. This can be either a string or a list.
        """
        if isinstance(key, str):
            return int(key.split("~")[0][2:])
        elif isinstance(key, list):
            if not isinstance(key[0], int):
                raise TypeError("If key is a list, then the first element must be an integer node id number.")
            return key[0]
        else: raise TypeError(f"key must be a string or a list, not {type(key)}!")

    def _specificity_level(self, key_list: list[int | str]):
        """
        Identifies the specificity level of the key.
        
        The specificity level can be 'attribute', 'nodeid', or 'player'.
        
        Arguments
            • key_list: list[int | str]; A list representation of the key.  All
                list elements must be integers or a string attribute at the end.
            
        Returns: 
            • str; 'attribute', 'nodeid', or 'player'
        """

        if isinstance(key_list, str):
            "Assumes that the key is already properly formatted."
            if key_list.endswith("b:"): return "player"
            if "~" in key_list:
                if key_list.endswith("~"): return "nodeid"
                elif key_list.split("~")[-1] in self.supported_attributes:
                    return "attribute"
                raise ValueError(f"key {key_list} improperly formatted.  Possibly unsupported attribute.")
            else: raise ValueError(f"key {key_list} improperly formatted. Missing '~'.")

        if key_list[-1] in self.supported_attributes: 
            if not all(isinstance(ele, int) for ele in key_list[:-1]):
                raise TypeError("All elements in keys must be integers or a string attribute at the end.")
            if len(key_list) % 2 != 0:
                raise ValueError(f"Missing nodeid before attribute in {key_list}")
            return "attribute"

        for idx, ele in enumerate(key_list):
            numis = "nodeid" if idx % 2 == 0 else "player"
            if not isinstance(ele, int):
                raise TypeError(f"{numis}s must be integers, not {ele}.")
            if numis == "nodeid":
                if ele not in self.tree.nlst:
                    raise ValueError(f"Node id {ele} must be within the list of nodes in the tree: {self.tree.nlst}.")
            elif numis == "player":
                if not (0 <= ele <= self.tree.nplayers):
                    raise ValueError(f"Player number {ele} not in this {self.tree.players}.")
                
        return "nodeid" if len(key_list) % 2 != 0 else "player"       

    def formatted_key(self, key: str | list[int | str]) -> str:
        """
        Ensures that the key is converted into a standard format.
        
        Arguments
            • key: str | list[int | str]; The key to be 
                formatted. This can be a string or a list.
            
        Returns: 
            • str; The formatted key.
        """

        if key in self.supported_attributes:
            "This is the lazy way to enter a key."
            key = f"@n{self.nodeid}~{key}"

        if isinstance(key, list):      
            
            specificity = self._specificity_level(key_list=key)

            key_str = ""
            for idx, ele in enumerate(key):
                if idx < len(key) - 1:
                    if idx % 2 == 0:
                        key_str += f"@n{ele}~"
                    else: key_str += f"p{ele}b:"

            if specificity == "attribute":
                key_str += key[-1]
    
            return key_str
        
        elif isinstance(key, str):
            "Error checking the string format."
            if "b" not in key:
                splitkey = key.split("~")
                if len(splitkey) == 2 and splitkey[0][2:].isdigit() and \
                    splitkey[-1] in self.supported_attributes: return key
                raise ValueError("key must contain 'b:...' meaning 'believes that...'")
            if key[0] == "p":
                key = f"@n{self.nodeid}~"
            elif key[:2] != "@n":
                raise ValueError("key must begin by specifying the node it is at via '@n...'")

            attribute = key.split("~")[-1]

            if attribute not in self.supported_attributes:
                raise ValueError(f"Supportes attributes: {self.supported_attributes}, not {attribute}!")
            
            key = "".join(key.split("~")[:-1])
         
            "Adding missing delimiters "
            key = re.sub(r'b(?!:)', 'b:', key)
            key = re.sub(r'(?<!~)p', '~p', key)
            key = re.sub(r'(?<!@)n', '@n', key)

            key += f"~{attribute}"

            if key[-1] == "~": key += "attribute"

            digits = "".join(key.split("~p")[:-1])
            digits = digits.replace("@n", "")
            digits = digits.replace("b:", "")
  
            if not digits.isdigit():
                for dig in digits:
                    if not dig.isdigit():
                        raise ValueError(f"Identified invalid character within key: {dig}")

            if isinstance(self.tree, Tree):
                for idx, dig in enumerate([int(item) for item in re.split(r'b:@n|~p', key)[1:-1]]):
                    if idx % 2 != 0:
                        if dig > self.tree.nlst[-1]:
                            raise ValueError(f"Node id in key: {dig} exceeds maximum node id number in Tree: {self.tree.nlst[-1]}!")
                    else:
                        if dig > self.tree.nplayers:
                            raise ValueError(f"Player number in key: {dig} exceeds number of players in Tree: {self.tree.nplayers}!")                        

            key_items = re.split(r'b:@n|~p', key)[1:-1]
            if "" in key_items:
                raise ValueError("Missing node id or player number detected in key!")

            return key

    def _attrinkey(self, key: str) -> str:
        """
        Returns the attribute present in the key, if any.
        
        Arguments
            • key: str; The key in string format.
            
        Returns: 
            • str; The attribute in the key if present, empty string otherwise.
        """
        if isinstance(key, str) and "~" in key:
            if key.split("~")[-1] in self.supported_attributes:
                return key.split("~")[-1]
        elif isinstance(key, list):
            if key[-1] in self.supported_attributes:
                return key[-1]
        return ""

    def keytolist(self, key: str) -> list[int | str]:
        """
        Converts a key into a list representation.
        
        For example, a key like '@n0~p1b:@n4~payoffs' will be converted into [0, 1, 4, 'payoffs'].
        
        Arguments
            • key: str; The key in string format.
            
        Returns: 
            • list[int | str]; The key in list format.
        """
        key = self.formatted_key(key)

        attr = self._attrinkey(key)
        
        key_items = [int(key.split("~")[0][2:])]
        for item in re.split(r'b:@n|~p', key[:-len(attr) - 1])[1:]:
            if item.isdigit(): key_items.append(int(item))
            else:
                ending = item.split("~")
                key_items.append(int(ending[0]))
                key_items.append(ending[-1])

        if bool(attr): key_items.append(attr)

        return key_items

    def _base_case_key(self, key: str) -> bool:
        """
        Returns if the key is a base case, meaning that it refers to an objective attribute 
        (without beliefs involved).
        
        Arguments
            • key: str; The key in string format.
            
        Returns: 
            • bool; True if the key is a base case, False otherwise.
        """
        if not "@n" in key or not "~" in key or key.endswith("b:"):
            raise ValueError(f"Improperly formatted key: {key}")

        splitkey = key.split("~")
        attr = splitkey[-1]

        if attr not in self.supported_attributes:
            print(f"key: {key} - attribute: {attr}")
            raise ValueError(f"key must contain one of these supported attributes: {self.supported_attributes}")
      
        if "b:" in key: return False
        
        if len(splitkey) == 2 and splitkey[0][-1].isdigit():
            return True

        return False

    def reduce_belief_level(self, key: str) -> str:
        """
        Alters the key such that the item may be found elsewhere. Belief dictionaries prevent a 
        combinatorial explosion of beliefs about beliefs by assuming that beliefs are inherited 
        from parents and that higher-order beliefs resemble lower-order beliefs.
        
        Arguments
            • key: str; The key in string format.
            
        Returns: 
            • str; The trimmed key.
        """
        if not isinstance(key, (str, list)):
            raise TypeError(f"key must be a string, not a {type(key)}!")        

        key = self.formatted_key(key)

        if self._base_case_key(key): return key

        if self._specificity_level(key) != "attribute":
            raise Exception(f"reduce_belief_level works only on attribute keys, not {self._specificity_level(key)} keys!")
  
        keylst = self.keytolist(key)

        if len(keylst) >= 6 and all(isinstance(ele, int) for ele in keylst[:5]) and keylst[0] == keylst[2] and keylst[1] == keylst[3]:
            """If the nested belief is redundant, then remove the redundancy.  For example, if at node 6 player 
            1 believes that at node 6 player 1 believes X, then simplify this to at node 6 player 1 believes X."""
            return self.formatted_key(keylst[2:])

        current_node = keylst[0]
        if current_node > 0:
            parent_node = self.tree.nodes[current_node].parent
            keylst[0] = parent_node

            return self.formatted_key(key=keylst)

        return self.formatted_key(key=keylst[2:])
        

class Node(dict):
    """
    Represents a node in a game tree.

    This class inherits from Python's built-in dictionary class, and thus the objects of the class can be treated as 
    dictionaries. This is done to make it easier to convert Node objects into JSON format. Each Node object stores the
    various properties and attributes related to that node. These properties include, but are not limited to, node ids,
    labels, parent identifiers, position, time, probability, information sets, choosers, choices, predictors, predictions, 
    payoffs, options and root. The class has methods to manage and manipulate these attributes 
    as per the requirements of the game theory.

    Each Node object can also have its own belief system, represented by a Beliefs object. This allows nodes to have
    different information from other nodes and can make different decisions based on their own beliefs. 
    """
    def __init__(self, nodeid: int, parentid: int, 
        label: str, level: int, nplayers_node: int = 2, root: 'Tree' = None):
        """
        Arguments
            • nodeid: int; The unique identifier for the node.
            • parentid: int; The id of the parent node.
            • label: str; The label for the node.
            • level: int; The level in the tree hierarchy (0 for root, 1 for root's children, etc.).
            • nplayers_node: int; The number of players for this node.
            • root: Tree; The root tree that the node belongs to.

        Attributes:
            • idnum: int; Unique identifier for the node.
            • label: str; Unique alphabetical and hierarchical label for the node.
            • level: int; Level of the node in the tree.
            • parent: int; Identifier for the parent node.
            • time: list[number, number]; Starting and ending timestamps when responses can be submitted.
            • probability: list[float, float]; First and second order probabilities.
            • positionxy: list[float, float]; Position of the node in a 2D layout.
            • info_set: list[list[int]]; Information set structure the node belongs to.
            • chooser: list[bool]; Indicates which players are choosers at the node.
            • choice: list[response_type]; Choices made by the players at the node.
            • predictor: list[bool]; Indicates which players are predictors at the node.
            • prediction: list[response_type]; Predictions made by the players at the node.
            • payoffs: list[int]; Payoffs for each player at the node.
            • _nplayers_node: int; Internal attribute to manage the number of players at the node.
            • options: list[Node]; List of subsequent nodes (options) that can be reached from the current node.
            • _root: Tree; The Tree object to which the node belongs.
            • beliefs: Beliefs; Belief system associated with the node.
        """       
        self.idnum: int = nodeid
        self.label: str = label
        self.level: int = level
        self.parent: int = parentid
        self.time: list[number, number] = [0, 0]
        self.probability: list[float, float] = [1.0, 1.0]
        self.positionxy: list[float, float] = [0.5, 1.0]
        self.info_set: list[list[int]] = [[nodeid]]
        self.chooser: list[bool] = [False] * nplayers_node
        self.choice: list[response_type] = [None] * nplayers_node
        self.predictor: list[bool] = [False] * nplayers_node
        self.prediction: list[response_type] = [None] * nplayers_node
        self.payoffs: list[int] = [0] * nplayers_node
        self._nplayers_node: int = nplayers_node
        self.options: list[Node] = []
        self._root: Tree = root

        self.beliefs = Beliefs(nodeid=self.idnum, tree=self._root)

        attrs_lst_node = ['idnum', 'parent', 'label', 'payoffs', 'chooser', 'choice', 'predictor', 
                          'prediction', 'probability', 'positionxy', 'time', 'info_set', 'beliefs', 'options']
        for attr in attrs_lst_node: self[f'{attr}'] = eval(f'self.{attr}')

    def isleaf(self) -> bool:
        """Returns if the node is a terminal node."""
        return not bool(self.options)

    def _update_list_attributes(self, new_nplayers: int) -> None:
        """Ensures that all list attributes are of equal length, which is the number of players on the tree.
        List attributes include chooser, choice, predictor, prediction, and payoffs.  Their lengths must equal
        the number of players because their elements correspond to the ith player.  Whenever the length of a 
        list attribute on any node changes, the attribute setter for that attribute changes nplayers_node, then
        the nplayers_node setter calls _update_list_attributes, which incriments or decriments the lengths of all
        the list attributes throughout the tree.

        Arguments
            • new_nplayers: int; The new number of players on the tree.
        """
        if new_nplayers != self._nplayers_node:
            defaults = [False, None, False, None, 0]
            attributes = ['chooser', 'choice', 'predictor', 'prediction', 'payoffs']
            for attr, default in zip(attributes, defaults):
                attribute = getattr(self, attr)
                if attribute is not None:
                    if len(attribute) < new_nplayers:
                        setattr(self, attr, attribute + [default] * (new_nplayers - len(attribute)))
                    elif len(attribute) > new_nplayers:
                        setattr(self, attr, attribute[:new_nplayers])

            self._nplayers_node = new_nplayers

    @property
    def nplayers_node(self):
        """
        Returns the number of players at this node.  This 
        is made to match the number of players in all nodes.
        """
        return self._nplayers_node

    @nplayers_node.setter
    def nplayers_node(self, value: int):
        """
        Sets the number of players in the node and updates all the 
        list attributes in the tree.

        Arguments
            • value: int; The new number of players on the tree.
        """
        self._update_list_attributes(new_nplayers=value)
        if self._root is not None: 
            self._root.nplayers = [value, self.idnum]
            self._root['nplayers'] = value

    @property
    def chooser(self):
        """
        Returns the believed state of 'chooser' attribute which is a list
        of booleans indicating which players are choosers at this node.
        """
        return self.believed("chooser")

    @chooser.setter
    def chooser(self, value: list[bool]):
        """
        Sets the chooser attribute which is a list of booleans 
        indicating which players are choosers at this node. 
        Sets the predictor attribute such that if a player 
        is either a chooser or a predictor but not both.

        Arguments
            • value: list[bool]; A list of booleans indicating the chooser status of players.
        """
        if not isinstance(value, list) or any(not isinstance(element, bool) for element in value):
            raise AttributeError(f"chooser attribute must be a list of bools, not {value}")
        
        self._chooser = value

        "Update the predictor attribute based on the chooser attribute"
        self._predictor = [not chooser if any(self._chooser) else False for chooser in self._chooser]
        self.nplayers_node = len(value)
        if self._root and self._root.memo:
            self._root.memo.cache = {}

    # @property
    # def choice(self):
    #     """
    #     Returns the choice attribute which is a list of choices made by the players.
    #     """
    #     return self._choice

    # @choice.setter
    # def choice(self, value: list[response_type]):
    #     """
    #     Sets the choice attribute and calls update_current_nodeid if all players have finished responding.

    #     Arguments
    #         • value: list[response_type]; A list of responses chosen by the players.
    #     """
    #     "Checking for errors in the new choice attribute."
    #     if not isinstance(value, (list, tuple)):
    #         raise ValueError("Choice attributes must be lists or tuples.")
        
    #     if isinstance(value, tuple):
    #         "update_current_nodeid is solidifying responses into an immutable type to prevent overwriting."
    #         self._choice = value
    #     else:
    #         if any(self.chooser):
    #             "If this is a choice node"
    #             for idx, (chooser, choice) in enumerate(zip(self.chooser, value)):
    #                 if not chooser and choice is not None:
    #                     violation_message = f"Player {idx} is not a chooser at node {self.idnum}"
    #                     violation_message += f" and so they cannot apply a choice there."
    #                     raise ValueError(violation_message)
                    
    #                 if chooser and not self.valid_response(response=choice):
    #                     raise ValueError(f"Improperly formatted choice: {choice}!")
            
    #         if not self.choicetypeis("simultaneous"):
    #             for choice in value:
    #                 if choice is not None:
    #                     if len(choice["option"]) != self.level + 1:
    #                         raise ValueError("The length of the option label of the chosen node must be equal to the level of the chosen node!")

    #         self._choice = value
    #         self.nplayers_node = len(value)

    #         if not self.isleaf() and self.finished_responding() and self._root is not None:
    #             "Automatically updating current node id if and when all responses are applied."
    #             self._root.update_current_nodeid()          

    @property
    def predictor(self):
        """
        Returns the predictor attribute which is a list of booleans 
        indicating which players are predictors at this node.
        """
        return self._predictor

    @predictor.setter
    def predictor(self, value: list[bool]):
        """
        Sets the predictor attribute and calls update_current_nodeid if all players have finished responding.

        Arguments
            • value: list[response_type]; A list of responses chosen by the players.
        """
        self._predictor = value
        self.nplayers_node = len(value)

    # @property
    # def prediction(self):
    #     """
    #     Returns the prediction attribute which is a list of predictions made by the players.
    #     """
    #     return self._prediction

    # @prediction.setter
    # def prediction(self, value: list[response_type]):
    #     """
    #     Sets the prediction attribute which is a list of predictions made by the players.

    #     Arguments
    #         • value: list[response_type]; A list of predictions made by the players.
    #     """

    #     "Checking for errors in the new prediction attribute."
    #     if not isinstance(value, (list, tuple)):
    #         raise ValueError("Prediction attributes must be lists or tuples.")
        
    #     if isinstance(value, tuple):
    #         "update_current_nodeid is solidifying responses into an immutable type to prevent overwriting."
    #         self._prediction = value
    #     else:
    #         if not self.choicetypeis("chance"):
    #             for idx, (predictor, prediction) in enumerate(zip(self.predictor, value)):
    #                 if not predictor and value[idx] is not None:
    #                     violation_message = f"Player {idx} is not a predictor at node {self.idnum}"
    #                     violation_message += f" and thus cannot apply a prediction there."
    #                     raise ValueError(violation_message)
                    
    #                 if predictor and not self.valid_response(response=prediction):
    #                     raise ValueError(f"Improperly formatted prediction: {prediction}!")
                
    #         self._prediction = value
    #         self.nplayers_node = len(value)

    #         if not self.isleaf() and self.finished_responding() and self._root is not None:
    #             "Automatically updating current node id if and when all responses are applied."
    #             self._root.update_current_nodeid()

    @property
    def payoffs(self):
        """
        Returns the believed state of the 'payoffs' attribute
        which is a list of payoffs corresponding to each player.
        """
        return self.believed("payoffs")

    @payoffs.setter
    def payoffs(self, value: list[int]):
        """
        Sets the payoffs attribute which is a list of payoffs 
        corresponding to each player.

        Arguments
            • value: list[int]; A list of integers indicating the payoffs of players.
        """
        self._payoffs = value
        self.nplayers_node = len(value)
        if self._root and self._root.memo:
            self._root.memo.cache = {}

    @property
    def probability(self):
        """
        Returns the believed state of the 'probability' attribute.
        """
        return self.believed("probability")

    @probability.setter
    def probability(self, value: list[float, float]):
        """
        Sets the probability attribute for the node, which is a list containing:
            1) the probability that this node will be chosen by a chance node (or ambdicated choice)
            2) the certainty in this probability, where 0.0 = totally ambigious and 1.0 = totally certain

        Arguments
            • value: list[float, float]; A list of floats between 0.0 and 1.0.
        """
        try:
            for idx, element in enumerate(value):
                if not isinstance(element, (float, int)) or idx > 1: 1 + "1"
        except: raise AttributeError("probability attribute must be a list of numbers like [probability, certainty].")
        
        prob, certainty = value
        if not (0 <= prob <= 1) or not (0 <= certainty <= 1):
            raise AttributeError("probability and certainty must be between 0 and 1.")
        
        self._probability = value

        if self._root and self._root.memo:
            self._root.memo.cache = {}

    @property
    def abdication_penalty(self):
        """
        The payoff penalty for abdicating one's response.
        """
        penalty = 10
        experiment = self._root.experiment
        if experiment and hasattr(experiment, 'experiment_configuration_dict'):
            settings: dict = self._root.experiment.experiment_configuration_dict
            payoff_settings: dict = settings.get('payoff_dimensions', {})
            if payoff_settings:
                penalty = payoff_settings.get('abdication_penalty', penalty)

        return penalty

    def believed(self, attr: str):
        """
        Retrieves the believed value of an attribute from the perspective of a specific player.

        Arguments
            • attr: str; Name of the attribute to retrieve.

        Returns
            • Any; Believed value of the specified attribute.
        
        Note: This function expects 'attr' to be in tree.beliefs.supported_attributes.
        """
        if self._root is not None and self._root.nodes is not None and self._root.perspective_of_player is not None:
            believed_attr = self.beliefs[f"{self._root.perspective_of_player}@n{self.idnum}~{attr}"]
            if believed_attr: return believed_attr        
        return eval(f"self._{attr}")

    def choicetypeis(self, choicetype: str | None = None) -> str | bool:
        """
        Determines the type of node based on the chooser attribute.

        The chooser attribute is a list of booleans.
            • Chance nodes: All are False
            • Sequential nodes: Only one is True
            • Simultaneous nodes: More than one are True        

        Arguments
            • choicetype: str | None; Expected node type ('chance', 'sequential', or 'simultaneous'). 
                If None, the function just returns the type.

        Returns
            • str | bool; If choicetype is None, returns a string ('chance', 'sequential', or 'simultaneous'). 
                If choicetype is specified, returns True if the node type matches the given type, False otherwise.
        """
        if not any(self.chooser): the_type_is = "chance"
        elif sum(bool(chooser) for chooser in self.chooser) == 1: 
            the_type_is = "sequential"
        else: the_type_is = "simultaneous"
        
        if choicetype is None: return the_type_is
        return choicetype == the_type_is

    def draw_probability(self) -> float:
        """
        Randomly samples the probability if the node's probability is ambiguous.

        Returns
            • float; The sampled probability.
        """
        if self.probability[1] == 1: return self.probability[0]
        if self.probability[1] == 0: return round(random.random(), 4)
        if self.probability[0] == 0.5: return round(random.uniform(\
            0.5 - (1 - self.probability[1]) / 2, 0.5 + (1 - self.probability[1]) / 2), 4)
        ambiguity_interval_lower = self.probability[0] - (1 - self.probability[1]) / 2
        ambiguity_interval_upper = self.probability[0] + (1 - self.probability[1]) / 2
        if ambiguity_interval_lower < 0: 
            ambiguity_interval_upper -= ambiguity_interval_lower
            ambiguity_interval_lower = 0
        elif ambiguity_interval_upper > 1: 
            ambiguity_interval_lower -= ambiguity_interval_upper - 1
            ambiguity_interval_upper = 1
        return round(random.uniform(\
            ambiguity_interval_lower, ambiguity_interval_upper), 4)

    @staticmethod
    def valid_response(response: response_type) -> bool:
        """
        Checks if a response is properly formatted.

        Arguments
            • response: dict; The response to be checked.

        Returns
            • bool; True if the response is properly formatted, False otherwise.
        """
        if response is None: return True

        if not isinstance(response, dict):
            return False
        
        if "option" in response and "rtimedn" in response and "timestamp" in response and len(list(response.keys())) < 6:
            "This is a response already applied to a tree, which does not need metadata."
            return True

        required_data_keys = {'option', 'keypress', 'rtimedn', 'rtimeup', 'timestamp'}
        required_metadata_keys = {'nodeid', 'round_room_batch', 'player_uuid', 'player_num'}

        if not ('data' in response and 'metadata' in response):
            return False
        
        data_keys = set(response['data'].keys())
        metadata_keys = set(response['metadata'].keys())

        return data_keys.issuperset(required_data_keys) and metadata_keys.issuperset(required_metadata_keys)

    def _responder_number(self, response_type: str = "choice") -> list[int]:
        """
        Retrieves the player numbers for the chooser(s) or predictor(s) at this node.

        Arguments
            • response_type: str; The type of response to consider ('choice' or 'prediction').

        Returns
            • list[int]; A list of player numbers.
        """
        if self.choicetypeis("chance"): return [-1]

        if response_type == "choice":
            return [player_index for player_index, chooser_bool in enumerate(self.chooser) if chooser_bool]
        
        elif response_type == "prediction":
            return [player_index for player_index, predictor_bool in enumerate(self.predictor) if predictor_bool]
        
        else: raise ValueError(f"Invalide response_type {response_type}. Use 'choice' or 'prediction'.")

    def _chooser_number(self) -> int:
        """
        Determines the player number of the chooser at this node.

        Returns
            • int; The player number of the chooser.
        """
        chooser_numbers = self._responder_number("choice")

        if len(chooser_numbers) == 1: 
            return chooser_numbers[0]
        else:
            "If simultaneous choice, return the primary chooser."
            pyramid_level = 0
            for pyramid_level_, pyramid_layer in enumerate(self.info_set):
                if self.idnum in pyramid_layer: 
                    pyramid_level = pyramid_level_
                    break

            while pyramid_level >= len(chooser_numbers):
                "Prevent index error."
                pyramid_level -= 1

            return chooser_numbers[pyramid_level]                  

    def _formatted_belief_key_prefix(self, belief_key_prefix: str) -> str:
        """
        Checks and corrects the format of the belief_key_prefix.

        Arguments
            • belief_key_prefix: str; The belief_key_prefix to format.

        Returns
            • str; The formatted belief_key_prefix.
        """
        if belief_key_prefix is None:
            belief_key_prefix = ""

        if not isinstance(belief_key_prefix, str):
            raise ValueError(f"belief_key_prefix type {type(belief_key_prefix)} ≠ string.")

        for unnecessary_character in ["@n", "~p", "~"]:
            belief_key_prefix = belief_key_prefix.replace(unnecessary_character, "")
        
        if belief_key_prefix == "":
            "If the prefix is non-specific, use the uuid of the chooser."
            choice_type = self.choicetypeis()
            if choice_type == "chance": return ""

            chooser_number = self._chooser_number()

            uuid = self._root.players[chooser_number]['uuid']
            return f"{uuid}believes:"
        
        "If the prefix is specific."
        if ":" not in belief_key_prefix:
            raise ValueError(f"{belief_key_prefix} must contain ':' delimiters.")
        
        belief_prefix = ""
        for believer in belief_key_prefix.split(":"):
            if believer != "":
                if believer.endswith("b"): believer_ = believer[:-1]
                elif believer.endswith("believes"): believer_ = believer[:-8]
                else: raise ValueError(f"belief_key_prefix {belief_key_prefix} must contain 'b:' or 'believes:'")

                if believer_.isdigit():
                    uuid = self._root.players[int(believer_)]['uuid']
                    belief_prefix += f"{uuid}believes:"   
                elif self._root.players_dict.agent_parameters.isuuid(key=believer_):
                    if believer_ not in [plr['uuid'] for plr in self._root.players]:
                        raise ValueError(f"{belief_key_prefix} refers to a player uuid not stored in {self._root.title}.")
                    belief_prefix += f"{believer_}believes:"     
                else: raise ValueError(f"Invalid character identified in belief_key_prefix: {believer_}") 

        return belief_prefix

    def _shorten_belief_key_prefix(self, belief_key_prefix: str) -> str:
        """
        Abbreviates the belief_key_prefix for easy reference.

        Arguments
            • belief_key_prefix: str; The belief_key_prefix to shorten.

        Returns
            • str; The shortened belief_key_prefix.
        """
        belief_key_prefix = self._formatted_belief_key_prefix(belief_key_prefix)
        short_prefix = ""
        for uuid in belief_key_prefix.split("believes:"):
            if uuid != "":
                short_prefix += f"{self._root.players_dict.agent_parameters.uuids.index(uuid)}b:"
        return short_prefix

    def _belief_depth(self, belief_key_prefix: str) -> int:
        """
        Determines the belief depth of the belief_key_prefix.

        Arguments
            • belief_key_prefix: str; The belief_key_prefix to check.

        Returns
            • int; The belief depth of the belief_key_prefix.
        """
        belief_key_prefix = self._formatted_belief_key_prefix(belief_key_prefix)
        return belief_key_prefix.count("believes:")

    def _softmax(self, utilities: list[float]) -> list[float]:
        """
        Transforms utilities into choice probabilities using the softmax function.

        Arguments
            • utilities: list[float]; The utilities to transform.

        Returns
            • list[float]; The transformed probabilities.
        """

        exp_choice_utilities = np.exp(utilities)
        denominator = np.sum(exp_choice_utilities)
        return [round(choice_utility / denominator, 6) 
                for choice_utility in exp_choice_utilities] 

    def expected_payoffs(self, belief_key_prefix: str = None, objective_probabilities: bool = True, maximize_utility: bool = False) -> list[float]:
        """
        Calculates expected payoffs at this node.
        
        Arguments
            • belief_key_prefix: string; The beginning of all keys used to access agent parameters within the Players dict.
                Example: '1b:2b:1b:' means that player 1 believes that player 2 believes that player 1 believes that....
            • objective_probabilities: bool; If True, this uses the dormant probabilities stored within the nodes to calculate
                expected payoffs.  Otherwise, this uses the subjective probabilities based on expectations of player choices.
            • maximize_utility: If True and not objective_probabilities, the agents always chooses the highest utility option. 

        Returns
            • list[float]; The expected payoffs indexed by player
        """
        
        if belief_key_prefix is None:
            "Initialize belief_key_prefix"
            belief_key_prefix = ""

        "Memoization key is a tuple of the method name and all arguments + the node id number"
        memo_key = ('expected_payoffs', belief_key_prefix, objective_probabilities, self.idnum)

        cached_result = self._root.memo.get(memo_key)
        if cached_result:
            return cached_result

        if self.isleaf():  
            "If the node is a terminal node, store the immediate payoff in memo"
            expected_payoffs = list(self.payoffs)

        else:
            "The player that is calling this method - the player who's perspective is taken"
            believer = belief_key_prefix.split("believes:")[0]

            "Reasoning depth parameter determine's a player's maximum recursion depth."
            reasoning_depth = int(self._root.players_dict[f"{believer}rdepth"]["mean"])

            "Will not incriment belief depth if this is a chance node or if the believer has maxed out their reasoning depth"
            if not (self.choicetypeis("chance") or reasoning_depth <= self._belief_depth(belief_key_prefix)): 
                "Incrimenting belief_key_prefix before recursion to reflect the beliefs of the chooser"
                chooser_uuid = self._root.players[self._chooser_number()]['uuid']
                next_order_belief = f"{chooser_uuid}believes:"
                if belief_key_prefix.endswith(next_order_belief):
                    "Removing redundant belief nesting: 'I believe that I believe...' -> 'I believe...'"
                    next_order_belief = ""
            else: next_order_belief = ""

            "Compute the probabilities of all the children of this node."
            choice_probabilities = self.choice_probabilities(\
                belief_key_prefix = "objective" if objective_probabilities else belief_key_prefix, 
                maximize_utility=maximize_utility)

            "Expected payoffs includes the immediate payoffs."
            expected_payoffs = list(self.payoffs)
            for option_index, option in enumerate(self.options):
                "Recursive call to calculate the expected payoffs of the option"
                option_expected_payoffs = option.expected_payoffs(\
                    belief_key_prefix + next_order_belief, 
                    objective_probabilities = objective_probabilities)
                
                for payoff_index, expected_payoff in enumerate(option_expected_payoffs):
                    "Incriment immediate payoffs by the expected payoffs."
                    expected_payoffs[payoff_index] += expected_payoff * choice_probabilities[option_index]

            expected_payoffs = [round(epayoff, 6) for epayoff in expected_payoffs]
            
        self._root.memo.set(memo_key, expected_payoffs)
        
        return expected_payoffs

    def payoff_dimensions(self, belief_key_prefix: str = None) -> dict[str: list[float]]:
        """
        Computes the payoff dimensions for this node.  

        Payoff Dimensions
        • intertemporal - change in expectation over time: 
            The expected payoff difference between node and parent.
            δπ_i(node X_t) = Eπ_i(node X_t) - Eπ_i(node X_(t-1))
        • interpersonal - payoff differences between players: 
            The payoff differences between players within this node.
            ∆π_i(node X_t) = π_i(node X_t) - π_j(node X_t)
        • stakes  - raw total payoffs: 
            Conventionally the first player's payoff. 
            π_i(node X_t)

        These three dimensions provide a complete description of binary dictator game payoff
        structures, meaning that the exact payoffs can be specified if these dimensions are
        known.  These dimensions activate social preferences, such as valuing the interests
        of self and other and motives concerning social comparison.
        
        Arguments:
            • belief_key_prefix - string: The beginning of all keys used to access agent parameters within the Players dict.
                Example: '1b:2b:1b:' means that player 1 believes that player 2 believes that player 1 believes that....

        Returns:
            • dict: Dictionary specifying each dimension.
                Example: {'intertemporal': [-3.0, 1.0], 'interpersonal': [6.0, 2.4], 'stakes': [5.0]}
        """       

        if self.idnum == 0:
            "Because the root node has no parent, default to the dimensions of the first born child node."
            try: return self.options[0].payoff_dimensions(belief_key_prefix)
            except IndexError: raise ValueError("It is pointless to compute the dimensions of a single-node tree.") 
        
        expected_payoffs_t = self.expected_payoffs(belief_key_prefix, objective_probabilities=True)
        expected_payoffs_t_minus_1 = self._root.nodes[\
            self.parent].expected_payoffs(belief_key_prefix, objective_probabilities=True)
       
        payoff_differences_intertemporal = list(np.round((np.subtract(\
            expected_payoffs_t, expected_payoffs_t_minus_1)), 4))        
        payoff_differences_interpersonal = list(np.round([payoff - np.average(\
            expected_payoffs_t) for payoff in expected_payoffs_t], 4))
       
        return {
            "intertemporal": payoff_differences_intertemporal, 
            "interpersonal": payoff_differences_interpersonal, 
            "stakes": [round(float(expected_payoffs_t[0]), 4)]
            }    

    def utility(self, player_number: int = None, belief_key_prefix: str = "") -> float:
        """Returns the utilities at this node for the player indicated by player_number.

        Arguments
            • player_number - int: The player number who's utility is of interest
            • belief_key_prefix - string: The beginning of all keys used to access agent parameters within the Players dict.
                Example: '1b:2b:1b:' means that player 1 believes that player 2 believes that player 1 believes that....
            
        Returns
            • float: The utility for the player at this node
        """
        
        if player_number == "all players":
            "This is a quick way to get the utilities for all players at this node."
            return [self.utility(plr, belief_key_prefix) for plr in range(self.nplayers_node)]

        elif player_number is None:
            "By default the player number is the chooser at the parent of this node."
            player_number = self._root.nodes[self.parent]._chooser_number()
    
        if not isinstance(player_number, int) or player_number < 0 or player_number > self.nplayers_node:
            raise ValueError(f"player_number {player_number} not in {self._root.title}.")

        def prospect_theory_utility(payoff_difference: int | float = None,
                                    payoff_outcome: int = None, payoff_reference: int = None, 
                                    risk_aversion: float = 0.8, loss_aversion: float = 1.2):
            """Inspired by Prospect Theory (Kahneman & Tversky, 1979), this return the utility
            derived from the payoff in the outcome and the payoff in the reference point.

            u(δπ_i) = δπ_i^r if δπ_i ≥ 0 else -l|δπ_i|^r
            where δπ_i = payoff difference for player i, 
            r = risk aversion, and l = loss aversion"""

            if payoff_difference is None:
                if payoff_outcome is None or payoff_reference is None:
                    raise ValueError("Either provide both payoffs or just the payoff difference.")
                payoff_difference = payoff_outcome - payoff_reference

            if payoff_difference >= 0:
                utils = payoff_difference ** risk_aversion
            else: utils = -loss_aversion * abs(payoff_difference) ** risk_aversion

            return utils

        belief_key_prefix = self._formatted_belief_key_prefix(belief_key_prefix)

        "Extracting the relevant payoff dimensions"
        payoff_dims = self.payoff_dimensions(belief_key_prefix)

        "Getting the uuid for the player"
        uuids = [plr['uuid'] for plr in self._root.players]
        plr_uuid = uuids[player_number]

        pairwise_values = [
            self._root.players_dict[f"{belief_key_prefix}{plr_uuid}values{uuids[playerj]}"]["mean"]
            for playerj in range(self.nplayers_node)
            ]
   
        individual_parameters = {
            param: self._root.players_dict[belief_key_prefix + plr_uuid + param]["mean"] 
            for param in ["risk", "loss", "yvne", "envy"]
            }
    
        Upds_term, Updo_term, Usod_term = 0, 0, 0
        for player_index, iVx in enumerate(pairwise_values):
            if player_index == player_number:
                iVi = iVx
                payoff_difference_self = payoff_dims["intertemporal"][player_number]
                Upds = prospect_theory_utility(payoff_difference=payoff_difference_self,
                                                risk_aversion=individual_parameters["risk"], 
                                                loss_aversion=individual_parameters["loss"])
                Upds_term += iVi * Upds
               
            else:
                iVj = iVx
                payoff_difference_other = payoff_dims["intertemporal"][player_index]
                Updo = prospect_theory_utility(payoff_difference=payoff_difference_other,
                                                risk_aversion=individual_parameters["risk"], 
                                                loss_aversion=individual_parameters["loss"])
                Updo_term += iVj * Updo
          
        self_other_difference = payoff_dims["interpersonal"][player_number]
        Usod = prospect_theory_utility(payoff_difference=self_other_difference,
                                        risk_aversion=individual_parameters["risk"], 
                                        loss_aversion=individual_parameters["loss"])

        if self_other_difference >= 0:
            Usod_term -= individual_parameters["yvne"] * Usod
        else: Usod_term += individual_parameters["envy"] * Usod

        utility = round(Upds_term + Updo_term + Usod_term, 6)
        
        return utility

    def choice_probabilities(self, belief_key_prefix: str = "", maximize_utility: bool = True) -> list[float]:
        """
        Computes and returns the probabilities that the choosing player(s) at this node will choose each option.
        
        This method uses the softmax function to convert the utilities of each option into probabilities. For 
        chance nodes, it returns the probabilities of the chance outcomes, which are independent of utilities. 
        If the belief_key_prefix argument is set to "objective", it will return the objective probabilities 
        instead of choice probabilities. 

        Arguments
            • belief_key_prefix: A string that indicates the perspective from which these probabilities 
                are being computed. It represents the chain of beliefs about beliefs leading to this 
                perspective.  For example, '1b:2b:1b:' means that player 1 believes that player 2 
                believes that player 1 believes....
            • maximize_utility: If True, the agents always chooses the highest utility option.

        Softmax Function
            p_i(choose X) = e^(U_i(option X)) / (e^(U_i(option X)) + e^(U_i(option Y)))                            

        Returns
            • list[float]; A list of probabilities for each option at this node.
        """
        if self.isleaf(): 
            raise Exception(f"choice_probabilities cannot be called on leaf nodes, like node {self.idnum}")
        
        if self.choicetypeis("chance") or belief_key_prefix == "objective":
            "At chance nodes, probabilities are objective and independent of utilities."
            return [option.draw_probability() for option in self.options]

        "Ensuring that the belief_key_prefix is properly formatted"
        belief_key_prefix = self._formatted_belief_key_prefix(belief_key_prefix)

        if self.choicetypeis("simultaneous"):
            return self.simultaneous_choice_probabilities(belief_key_prefix=belief_key_prefix)
    
        "Player number of the chooser at this node"
        chooser_number = self._chooser_number()

        "Calculating the utilities of all children of this node"
        option_utilities = [
            option.utility(player_number=chooser_number, belief_key_prefix=belief_key_prefix) 
            for option in self.options
            ]
        
        if maximize_utility:
            "Return deterministic choice probabilities, always choosing the highest utility option"
            max_utility = max(option_utilities)
            num_max_utilities = option_utilities.count(max_utility)
            return [
                1 / num_max_utilities if utility == max_utility else 0
                for utility in option_utilities
            ]            

        "Convert utilities into choice probabilities"
        return self._softmax(option_utilities)

    def simultaneous_choice_probabilities(self, belief_key_prefix: str) -> list[float]:
        """
        Computes and returns the probabilities of the options being chosen in a simultaneous choice node.

        This method is used when players must commit to a choice before knowing the choices of other players. 
        It simulates recursive Theory of Mind reasoning, where each player considers what other players will 
        likely choose, and what others believe they will choose.

        Arguments
            • belief_key_prefix: A string that indicates the perspective from which these probabilities are being 
                            computed. It represents the chain of beliefs about beliefs leading to this perspective.

        Returns
            • list[float]: A list of probabilities for each option at this node.
        """
        if not self.choicetypeis("simultaneous"):
            raise Exception("simultaneous_choice_probabilities must be used on simultaneous choice nodes.")
        
        if "believes:" not in belief_key_prefix:
            raise Exception("belief_key_prefix in simultaneous_choice_probabilities should indicate the perspective taken.")

        "Memoization key is a tuple of the method name and all arguments + the node id number"
        memo_key = ('simultaneous_choice_probabilities', belief_key_prefix, self.idnum)

        cached_result = self._root.memo.get(memo_key)
        if cached_result:
            return cached_result

        apex_nodeid = self.info_set[0][0]
        if not apex_nodeid == self.idnum:
            "If this is not an apex node, call this method at the apex."
            self._root.nodes[apex_nodeid].simultaneous_choice_probabilities(belief_key_prefix=belief_key_prefix)
        
        def simultaneous_probs(probability_dict: dict, utility_dict: dict, options_per_chooser: dict, chooser_idx: int, reasoning_depth: int) -> dict[str: float]:
            """Simulates the process of recursive Theory of Mind reasoning
            
            The choice probabilities are initialized to the dormant probabilities stored in each node.  With each recursive step,
            a player overwrites their probabilities with choice probabilities given the believed choice probabilities of other players.
            For instance, if in 'rock-paper-scissors' I think that you will play scissors, then I'll probably play rock.  If I think 
            you think this, then I think that you will play paper, so then I should play scissors instead.  But wait, if you think 
            this way, then I should play rock...  This process continues until the player calling this method maxes out their ability
            to think recursively."""

            if reasoning_depth <= 0: return probability_dict

            option_utilities = []
            my_options = options_per_chooser[chooser_idx]
            nchoosers = len(list(options_per_chooser.keys()))

            for label in my_options:
                option_utility = 0.0
                for permutation_key in utility_dict:
                    remaining_probs = 1.0
                    for olabel in permutation_key:
                        if olabel != label:
                            remaining_probs *= probability_dict[olabel]
                    option_utility += utility_dict[permutation_key][chooser_idx] * remaining_probs        
                option_utilities.append(option_utility)

            "Convert utilities of options into the probabilities that they will be chosen"
            updated_probabilities = self._softmax(option_utilities)

            "Enter updated probabilities into the probability dictionary"
            for olabel, probability in zip(my_options, updated_probabilities):
                probability_dict[olabel] = probability
            
            return simultaneous_probs(probability_dict = probability_dict, utility_dict = utility_dict, options_per_chooser = options_per_chooser,
                                chooser_idx = (chooser_idx + 1) % nchoosers, reasoning_depth = reasoning_depth - 1)

        "List of choosers involved in simultaneous choice"
        choosers = [player_index for player_index, chooser in enumerate(self.chooser) if chooser]
        
        "Stores the choice probabilities within information set structure.  Will be iteratively rewritten."
        probability_dict = {}
        for layer in self.info_set:
            for child in self._root.nodes[layer[0]].options:
                probability_dict[child.label] = child.probability[0]

        "The labels of each option for each chooser"
        option_labels_per_chooser = {
            chooser: [label for label in probability_dict.keys() if len(label) - self.level - 1 == chooser_index]
            for chooser_index, chooser in enumerate(choosers)
            }

        "All permutations of option labels per chooser"
        option_label_permutations = list(it.product(*list(option_labels_per_chooser.values())))

        "Node id numbers of all nodes at the base layer of the information set structure"
        base_level_nodeids: list[int] = [
            self._root.olst.index("".join(label[-1] for label in option_label_permutaion))
            for option_label_permutaion in option_label_permutations
            ]

        "Stores the utilities of all base layer nodes per chooser"
        utility_dict = {
            permutation: tuple([self._root.nodes[baseid].utility(
                player_number=chooser, belief_key_prefix=belief_key_prefix) for chooser in choosers])
            for baseid, permutation in zip(base_level_nodeids, option_label_permutations)
            }

        "The player that is calling this method - the player who's perspective is taken"
        believer = belief_key_prefix.split("believes:")[0]

        "Reasoning depth parameter determine's a player's maximum recursion depth."
        reasoning_depth = int(self._root.players_dict[f"{believer}rdepth"]["mean"])

        "Decrimenting reasoning depth by the belief depth indicated by the belief key prefix"
        reasoning_depth -= self._belief_depth(belief_key_prefix)

        "The player number of the first chooser at the apex node"
        first_chooser = next((chooser_index for chooser_index, 
                              chooser in enumerate(self._root.nodes[apex_nodeid].chooser) if chooser), None)

        simultaneous_probabilities = simultaneous_probs(probability_dict=probability_dict, utility_dict=utility_dict, 
                                        options_per_chooser=option_labels_per_chooser, chooser_idx=first_chooser, reasoning_depth=reasoning_depth)

        "Saving a memo for all nodes in the information set structure"
        for layer_index, layer in enumerate(self.info_set[::-1]):
            layer_index -= len(self.info_set) - 1
            "Iterating in reverse order so that the last 'child_probabilities' will apply to the apex node"
            for nodeid in layer:
                olabels_at_level = [olabel for olabel in sorted(probability_dict.keys()) if len(olabel) == self.level + layer_index + 1]
                child_probabilities = [simultaneous_probabilities[olabel] for olabel in olabels_at_level]
                memo_key = ('simultaneous_choice_probabilities', belief_key_prefix, nodeid)
                if child_probabilities:
                    self._root.memo.set(memo_key, child_probabilities)

        if not child_probabilities:
            raise Exception(f"Warning: Empty simultaneous choice probabilities at node {self.idnum} of tree {self._root.title} with belief key prefix: {belief_key_prefix}")

        "The choice probabilities for the children of the apex node."        
        return child_probabilities        


    def cumulative_payoffs(self, abdication_penalty: int = None) -> list[int]:
        """
        Computes and returns the cumulative payoffs for all players if this node is reached in the game.

        This method sums up the payoffs at all ancestor nodes of this node, including the payoff at this node itself.

        Returns
            • list[int]: A list of cumulative payoffs for each player.
        """
        penalty = self.abdication_penalty if abdication_penalty is None else abdication_penalty

        cpayoffs = [0] * self.nplayers_node

        for nodeid in self._root.path(finish=self._root.current_nodeid):
            this_node = self._root.nodes[nodeid]   
            for playernum, (payoff, abdicated) in enumerate(zip(
                this_node.payoffs, this_node.players_who_abdicated())):
                cpayoffs[playernum] += payoff
                if abdicated:
                    cpayoffs[playernum] -= penalty

        return cpayoffs


    def incriment_cumulative_payoffs(self, abdication_penalty: int = None) -> None:
        """
        Adds payoffs at this node to the players' 'cumulative_payoffs' 
        attribute, minus the abdication penalty if applicable.
        """
        penalty = self.abdication_penalty if abdication_penalty is None else abdication_penalty

        for playernum, (payoff, abdicated) in enumerate(zip(self.payoffs, self.players_who_abdicated())):
            self._root.players[playernum]['cumulative_payoffs'] = round(self._root.players[playernum]['cumulative_payoffs'] + payoff, 2)
            if abdicated:
                self._root.players[playernum]['cumulative_payoffs'] = round(self._root.players[playernum]['cumulative_payoffs'] - penalty, 2)      


    def players_who_abdicated(self) -> list[bool]:
        """
        List of players that abdicated a response at this node.
        """
        players_abdicated = [False for player in self._root.players]

        is_choice_node = not self.isleaf() and not self.choicetypeis('chance')
        is_factual_node = self.idnum in self._root.timeline()

        if is_choice_node and is_factual_node:
            for player_index in range(self._root.nplayers):
                plr_response = self.choice[player_index] if self.chooser[player_index] else self.prediction[player_index]

                if plr_response is None:
                    if self.idnum != self._root.current_nodeid:
                        players_abdicated[player_index] = True
                else:
                    node_duration = self.time[1] - self.time[0]
                    reaction_time = plr_response['rtimeup']
                    if reaction_time > node_duration:
                        players_abdicated[player_index] = True

        return players_abdicated


    def finished_choosing(self) -> bool:
        """
        Checks if the chooser(s) at this node have finished making their choices.

        This method checks whether valid responses have been recorded for all players that are choosers at this node.

        Returns
            • bool: True if all choosers have finished making their choices, False otherwise.
        """
        if self.isleaf(): return True
        if not any(self.chooser):
            "Handling chance nodes"
            for choice in self.choice:
                if choice is not None and self.valid_response(response=choice):
                    return True
            return False
        for chooser, choice in zip(self.chooser, self.choice):
            if chooser and (choice is None or not \
                self.valid_response(response=choice)): return False
        return True

    def finished_predicting(self) -> bool:
        """
        Checks if the predictor(s) at this node have finished making their predictions.

        This method checks whether valid predictions have been recorded for all players that are predictors at this node.

        Returns
            • bool: True if all predictors have finished making their predictions, False otherwise.
        """
        if self.isleaf(): return True
        if not any(self.chooser):
            "Handling chance nodes"
            for choice in self.choice:
                if choice is not None and self.valid_response(response=choice):
                    return True
            return False
        for predictor, prediction in zip(self.predictor, self.prediction):
            if predictor and (prediction is None or not \
                self.valid_response(response=prediction)): return False
        return True

    def finished_responding(self) -> bool:
        """
        Checks if all players at this node have finished responding, either by choosing or by predicting.

        NOTE 'valid_response' is redundant within finished_choosing and finished_predicting.
        TODO Remove this redundancy once the response system is thoroughly tested.

        Returns
            • bool: True if all choosers and predictors have finished responding, False otherwise.
        """
        return self.finished_choosing() & self.finished_predicting()

    def node_status(self) -> str:
        """
        Checks and returns the current status of this node.

        Returns
            • 'before': If no responses have been recorded.
            • 'during': If some responses have been recorded.
            • 'after':  If all responses have been recorded and the game can proceed to the next node.
        """
        if self.finished_responding(): return 'after'
        elif self.finished_choosing() or \
            self.finished_predicting(): return 'during'
        else: return 'before'

    def __setitem__(self, key, val) -> None:
        """
        Sets the value of an attribute of this node using the item assignment syntax (e.g., node[key] = val).

        Arguments
            • key: The name of the attribute.
            • val: The value to set the attribute to.
        """
        super(Node, self).__setitem__(key, val)
        try: setattr(self, f"_{key}", val)
        except: pass

    def __setattr__(self, name, value) -> None:
        """
        Sets the value of an attribute of this node using the attribute assignment syntax (e.g., node.name = value).

        Arguments
            • name: The name of the attribute.
            • value: The value to set the attribute to.
        """
        super(Node, self).__setattr__(name, value)
        if name.lstrip('_') in self: super(Node, self).__setitem__(name.lstrip('_'), value)

    def __getattr__(self, name):
        """
        Gets the value of an attribute of this node using the attribute access syntax (e.g., node.name).
        """
        if name == '_nplayers':
            return len(self.players)


class Tree(Node):
    """
    The Tree class is a core component of the Morality Game, representing moral and strategic dilemmas that participants 
    engage with. Game trees are the fundamental units of this game, dynamically storing response data and progressing the 
    game to further nodes as responses are collected. These trees are not just data structures, but dynamic, user-friendly 
    elements capable of evolving based on participant interaction.

    Each Tree is initiated with a title and allows for custom configuration such as the number of players, player details, 
    and associated experiment. The class features a series of methods and attributes to facilitate game progress and data 
    collection. In addition to standard game tree elements like nodes and edges, a tree also stores information about the 
    players and their perspectives.

    Some key features of the Tree class include:
    • Dynamic and Automatic Updates: Many of the Tree's attributes, such as the number of players or 
        player perspectives, can be updated on the fly. Changes automatically propagate to relevant 
        aspects of the tree, such as its nodes and edges.
    • Game Progression: The tree allows for the game to progress through its nodes as participants 
        make their moves. It also adjusts its status dynamically according to the game's progress.
    • Response Data Collection: Trees are not just about presenting dilemmas to participants, but 
        also about capturing their decisions. The response data is stored directly in the tree, 
        turning it into a data collection tool.
    • Versatility and User Friendliness: Trees can be created and adjusted ad hoc, accommodating 
        various game setups. This flexibility makes the Tree class adaptable to many scenarios.
    • JSON Compatibility: For easy interaction with the frontend, trees can emit JSON versions 
        of themselves, facilitating communication between the backend and participants.
    
    Using this class, one can create intriguing, engaging moral and strategic dilemmas for participants, 
    collect valuable data, and dynamically adjust to game progress. The Tree class thus plays a pivotal 
    role in the Morality Game, enabling a rich, interactive, and data-driven gaming experience.
    """
    def __init__(self, title, nplayers: int = 2, players: list[play.ParticipantsData.Player] = None, 
                 avatar_colors: list[tuple[int, int, int, float]] = None, adjacency_matrix: match.AdjacencyMatrix = None,
                 edges: list[int] = None, experiment: object = None, timestamps: dict[str: float | None] = None, 
                 randomize_positions = True, seconds_per_node = 12):
        """
        Initializes a Tree object. A tree represents a game tree, with 
        nodes representing game states and edges representing actions.
        
        Arguments:
            • title: str; The title or identifier of the tree.
            • nplayers: int; The number of players in the game. Default is 2.
            • players: list[play.ParticipantsData.Player]; A list of Player objects 
                representing the players. If None, default players will be generated.
            • avatar_colors: list[tuple[int, int, int, float]]; List of avatar colors as hsla tuples, 
                including the color of the chance node, which is maximally distinctive from the avatars.
            • adjacency_matrix: match.AdjacencyMatrix | None: Matrix of pairwise matching probabilities.     
            • edges: list[int]; A list of edges to add to the tree. If None, the tree will be edgeless.
            • experiment: object; An object representing the experiment associated with the game tree. 
                If None, no experiment will be associated.
            • randomize_positions: bool; If True, the node coodinates are randomized.
            • seconds_per_node: The number of seconds players will have to respond each node.
        """
        super().__init__(nodeid = 0, parentid = -1, label = '', level = 0, nplayers_node = nplayers, root = self)
        self.current_nodeid, self.title, self.round_room_batch, self.nodes = 0, title, (0, 0, 0, 0), {0: self, '': self}
        self.players: list[play.ParticipantsData.Player] = self.default_players(nplayers=nplayers, players=players)
        self.nplayers: int = len(players) if players is not None else nplayers
        self.nlst: list[int] = sorted([node for node in self.nodes if isinstance(node, int)])
        self.olst: list[str] = [self.nodes[node].label for node in self.nlst]
        self.levels: list[int] = [self.nodes[node].level for node in self.nlst]
        self.adjacency_matrix: match.AdjacencyMatrix = self.matching_probabilities(
            nplayers=nplayers) if adjacency_matrix is None else adjacency_matrix
        self.avatar_colors = avatar_colors if avatar_colors is not None else self._avatar_colors(tree=self)
        self.perspective_of_player: str = None
        self.memo = MemoizationCache()
        self.experiment = experiment
        
        self.timestamps: dict[str: float | None] = \
            timestamps if timestamps is not None else {
            'emit_tree_time': None,
            'update_node_time': None,
            'abdicate_node_time': None,
            'round_deadline_time': None,
            'round_started_time': None,
            'round_ended_time': None
        }
        
        attrs_lst_tree = ['adjacency_matrix', 'timestamps', 'players', 'avatar_colors', 
                          'round_room_batch', 'title', 'current_nodeid', 'nplayers']
        for attr in attrs_lst_tree: self[f'{attr}'] = eval(f'self.{attr}')

        if edges: self._edges_to_tree(edges=edges, randomize_positions=randomize_positions, seconds_per_node=seconds_per_node)

    def default_players(self, nplayers: int = 2, players: list[play.ParticipantsData.Player] = None) -> list[play.ParticipantsData.Player]:
        """
        Generates a list of default Player objects.
        
        Arguments:
            • nplayers: int; The number of players to generate. Default is 2.
            • players: list[play.ParticipantsData.Player]; A list of Player objects to 
                use instead of generating new ones. If None, new players will be generated.
            
        Returns:
            • list[play.ParticipantsData.Player]: A list of Player objects.
            
        Raises:
            • ValueError: If 'players' is not a list of Player objects.
        """
        if players is None: 
            "Generating maximally visually distinctive avatar colors"
            avatar_colors = []
            start_hue = random.randint(0, 359)
            for idx in range(nplayers):
                hue = int((start_hue + idx * (360 / (nplayers + 1))) % 360)
                satur, light = random.randint(50, 100), random.randint(35, 65)
                avatar_colors.append((hue, satur, light, 1.0))

            players_: play.ParticipantsData = play.ParticipantsData()
            for hsla_color in avatar_colors:
                players_.create_player(avatar_color=hsla_color)
            
            players = list(players_.values())

        elif isinstance(players, list) and all(isinstance(player, (int, str)) for player in \
            players) and self.experiment is not None and hasattr(self.experiment, "players_dict"):
            return [self.experiment.players_dict.get(player, None) for player in players]

        elif not isinstance(players, list) or any(not isinstance(player, 
                    play.ParticipantsData.Player) for player in players):
            try:
                players_ = []
                participants_data = play.ParticipantsData()
                for idx, player in enumerate(players):
                    participants_data.create_player(username = None, email_address = None, uuid_ = player['uuid'],
                      player_type=player["player_type"], avatar_color = player['avatar']['color'])
                    player = participants_data[idx]
                    players_.append(player)
                players = players_
            except: raise ValueError(f"players must be a list of Player instances.")

        return players

    @staticmethod
    def _avatar_colors(tree) -> list[tuple[int, int, int, float]]:
        """
        A list of all avatar colors + the chance node color as a list of hsla tuples
        """
        "Extract a list of avatar colors stored in the root of the tree."
        avatar_colors_ = [player['avatar']['color'] for player in tree.players]  

        avatar_colors = []
        for color in avatar_colors_:
            "Converting hsla strings into tuples"
            for char in ["hsla(", "%", ")"]:
                color = color.replace(char, "")
            color = color.split(", ")  

            avatar_colors.append(tuple([int(color[0]), int(color[1]), int(color[2]), round(float(color[3]), 2)]))  

        "Extract hue values"
        hues = [color[0] for color in avatar_colors]

        "Append first hue plus 360 (for circular hue space) to the end"
        hues.append(hues[0] + 360)

        "Find the largest gap"
        max_gap = max(hues[i + 1] - hues[i] for i in range(len(hues) - 1))
        
        "Find the mid-point of the largest gap"
        for idx in range(len(hues) - 1):
            if hues[idx + 1] - hues[idx] == max_gap:
                new_hue = (hues[idx] + hues[idx + 1]) // 2 % 360
                break

        avatar_colors.append((new_hue, 50, 50, 1.0))
        return avatar_colors

    @property
    def nplayers(self) -> int:
        """
        Returns the number of players in the tree.
        """
        return self._nplayers

    @nplayers.setter
    def nplayers(self, value):
        """
        Sets the number of players in the game. Updates all nodes 
        in the game tree and the adjacency matrix accordingly.
        
        Arguments:
            • value: int | list[int]; The new number of players. If a list, it should
                contain two elements: [new_number_of_players, node_id_to_exclude].
        
        Raises:
            • TypeError: If 'value' is not an integer or a list of integers.
        """
        if isinstance(value, list):
            self._nplayers = self['nplayers'] = value[0]
            except_node, value_ = value[1], value[0]
        else: self._nplayers, except_node, value_ = value, None, value
        
        if len(self._players) > value_: 
            self.players = self.players[:value_]
            self.adjacency_matrix = self.matching_probabilities(nplayers=self._nplayers)
            self.update_all_nodes_nplayers(new_nplayers=value_, except_node=except_node)
        elif len(self._players) < value_: 
            self.players += self.default_players(nplayers=value_+1)[len(self._players):]
            self.adjacency_matrix = self.matching_probabilities(nplayers=self._nplayers)
            self.update_all_nodes_nplayers(new_nplayers=value_, except_node=except_node)

    @property
    def players(self) -> list[play.ParticipantsData.Player]:
        """
        Returns a list of 'Player' dictionaries containing information about the players.
        """
        return self._players

    @players.setter
    def players(self, value):
        """
        Sets the list of players in the game. Updates the number of players.
        
        Arguments:
            • value: list[play.ParticipantsData.Player]; The new list of Player objects.
        
        Raises:
            • TypeError: If 'value' is not a list of Player objects.
        """
        self._players = value
        self.nplayers = len(value)
        if self.memo:
            self.memo.cache = {}

        # self.avatar_colors = self._avatar_colors()
        # self["avatar_colors"] = self._avatar_colors()

    @property
    def perspective_of_player(self) -> str | None:
        """
        Returns a belief key prefix representing the perspective taken upon this tree.
        Example: "@n7~p3b:", which means that the tree is being viewed from the perspective 
        of player 3 at node 7.  If None, the tree is being viewed objectively.
        """
        return self._perspective_of_player

    @perspective_of_player.setter
    def perspective_of_player(self, value):
        """
        Sets the perspective of the player for this game tree.
        
        Arguments:
            • value: int | str | list[int] | None; The new perspective. Can be an integer 
                (player number), a string (formatted perspective), a list (sequential 
                perspectives), or None (default perspective).
        
        Raises:
            • ValueError: If the perspective is improperly formatted.
            • TypeError: If 'value' is not an integer, string, list of integers, or None.
        """
        self._perspective_of_player = self._formatted_perspective(value)
        if self.memo:
            self.memo.cache = {}        

    def _formatted_perspective(self, value) -> str:
        """
        Converts the perspective input into a properly formatted belief key prefix. 
        
        Arguments:
            • perspective: Union[int, str, list[int], None]; Can be an integer (player number), a string
                (formatted perspective), a list (sequential perspectives), or None (default perspective).
        
        Returns:
            • str: A properly formatted belief key prefix.
        
        Raises:
            • ValueError: If the perspective is improperly formatted.
            • TypeError: If 'perspective' is not an integer, string, list of integers, or None.
            
        Example:
            For example, if the input perspective is [8,2,4,1], the output would be "@n8~p2b:@n4~p1b:", 
            which stands for "At node 8, player 2 believes that at node 4 player 1 believed that..."
        """
        if value is None: return None
        elif isinstance(value, int):
            "The tree is from the perspective of a particular player at the current node."
            if (0 <= value < self.nplayers):
                return f"@n{self._root.current_nodeid}~p{value}b:"
            else: raise ValueError(f"Player {value} not in tree {self.title}")
        elif isinstance(value, str):
            for ele in ["@n", "b:", "p", "~"]:
                if ele not in value:
                    raise ValueError(f"perspective_of_player must include {ele}")
            digits = str(value)    
            for ele in ["@n", "b:", "p", "~"]:
                digits = digits.replace(ele, "")
            for dig in digits:
                if not dig.isdigit():
                    raise ValueError(f"Improperly formatted perspective {value} - due to character {dig}")
            return value
        elif isinstance(value, list):
            "[nodeid, player_num, nodeid, player_num,...]"
            if not all(isinstance(ele, int) for ele in value):
                raise ValueError(f"All list elements must be integers when setting perspective via a list.")
            if len(value) % 2 != 0:
                raise ValueError(f"perspective must be formatted like [nodeid, player_num, nodeid, player_num,...]")
            new_val = ""
            for idx, ele in enumerate(value):
                if idx % 2 == 0: new_val += f"@n{ele}~"
                else: new_val += f"p{ele}b:"
            return new_val
        else: raise TypeError("perspective_of_player an be None, int, str, or list[int]")      


    def __getattr__(self, name):
        """
        Provides custom attribute access. If the attribute does not 
        exist, it tries to generate it from other existing attributes.
        
        Arguments:
            • name: str; The name of the attribute to access.
        
        Returns:
            • Various types: The value of the attribute, or a computed value if the attribute does not exist.
        """
        if name == 'nlst' and self.nodes is not None:
            "List of node 'idnum's"
            return [node for node in self.nodes]
        if name == 'olst' and self.nodes is not None:
            "List of node 'label's"
            return [self.nodes[node].label for node in self.nlst]
        if name == 'treetag':
            "Unique identifier of tree."
            self.tree_tag()
            return self.treetag
        if name == 'permutation':
            "Arrangement of nodes on computer screen."
            self.visual_permutation()
            return self.permutation
        if name == 'strategy_profiles':
            "Agents can save their precomputed responses here and use them to respond during a game."
            if 'strategy_profiles' not in self.__dict__:
                self.strategy_profiles = self.strategy_profile_maker()
            elif not self.memo.cache or len(self.strategy_profiles) != self.nplayers or len(self.strategy_profiles[0]) != self.nlst[-1]:
                "TODO 'strategy_profiles' should be updated every time the tree is significantly altered."
                self.strategy_profiles = self.strategy_profile_maker()
            return self.strategy_profiles
        if name == 'players_dict':
            "Instance of 'Players' dict stored at root node."
            if self.experiment is None:
                "If there is no experiment, create players_dict for testing purposes."
                if 'players_dict' not in self.__dict__:
                    self.players_dict = play.Players()
                    self.players_dict.add_players(number=self.nplayers)
                    for player_index, player in enumerate(self.players):
                        self.players_dict[player_index]['uuid'] = player['uuid']
                    self._root.players_dict.agent_parameters.uuids = [plr['uuid'] for plr in self._root.players]
                return self.players_dict   
            else: return self.experiment.players_dict             
        if name == 'agent_parameters':
            if self.experiment is None:
                return self.players_dict
            else: return self.experiment.players_dict
        if name == 'round_room_batch':
            current_node = self.nodes[self.current_nodeid]
            node_duration = current_node.time[1] = current_node.time[0]
            return (self.round_room_batch[0], self.round_room_batch[1], self.round_room_batch[2], int(time.time() + node_duration))
        

    def __repr__(self):
        """
        Pretty prints the tree by default.
        """
        return pp.pformat({key: val for key, val in self.items()}, sort_dicts=False)


    def __len__(self) -> int:
        """
        Returns the number of nodes in the tree.
        """
        return len(self.nlst)


    def update_all_nodes_nplayers(self, new_nplayers: int, except_node: int = None) -> None:
        """
        Updates the number of players in all nodes in the game tree, except for one.
        
        Arguments:
            • new_nplayers: int; The new number of players.
            • except_node: int; The ID of the node not to update. If None, all nodes will be updated.
        """
        if self.nodes is not None and except_node is not None:
            for idnum, node in self.nodes.items():
                if idnum != except_node:
                    node._update_list_attributes(new_nplayers=new_nplayers)


    def matching_probabilities(self, nplayers, distribution_type="uniform", coefficients=[.5, .5]) -> match.AdjacencyMatrix:
        """
        Generates a matching probabilities adjacency matrix between all players in the game tree.
        
        Arguments:
            • nplayers: int; The number of players.
            • distribution_type: str; The type of distribution to use for the probabilities. Default is "uniform".
            • coefficients: list[float]; The coefficients to use for the distribution. Default is [.5, .5].
        
        Returns:
            • match.AdjacencyMatrix: The generated adjacency matrix.
        """
        self.adjacency_matrix = match.AdjacencyMatrix(matrix=f"sample-{nplayers}", 
            distribution_type=distribution_type, coefficients=coefficients)
        return self.adjacency_matrix


    def tree_status(self) -> str:
        """
        Determines the status of the game: whether it has not started, is in progress, or has finished.
        
        Returns:
            • str: The status of the game. Can be "before" (game has not started), 
                "during" (game is in progress), or "after" (game has finished).
        """
        current_node = self.nodes[self.current_nodeid]
        if self.current_nodeid == 0: return 'before'
        elif current_node.isleaf() and self.nodes[self.nodes[\
            current_node.parent].info_set[0][0]].finished_responding(): return 'after'
        return 'during'


    """The following methods concern relationships between nodes: childrenof, parentof, siblingsof, 
    ancestorsof, descendantsof, birth_order, thisismy, sibling_groups, parent_groups, _common_ancestor, 
    path, and distance.  If trees are like family trees, many of these methods determine the family 
    relationships between nodes.  Typically these are helper methods used by other methods."""

    def childrenof(self, nodeid: int, depth: int = 1) -> list[int]:
        """
        Returns the children of a given node as identified by its node ID. If 'depth' is 
        greater than 1, it returns the grandchildren (or further descendants) instead.
        
        Arguments:
            • nodeid: int; The ID of the node whose children or descendants are to be found.
            • depth: int; The degree of descent from the node. Default is 1 (direct children).
        
        Returns:
            • list[int]: The IDs of the children or descendants of the node.
        
        Raises:
            ValueError: If 'depth' is less than 0.
        """
        if depth < 0: raise ValueError("Depth must be a non-negative integer.")
        if depth == 0: return [nodeid]
        node = self.nodes[nodeid]  
        children_ids = [child.idnum for child in node.options]
        if depth == 1: return children_ids
        else:
            grandchildren_ids = []
            for child_id in children_ids:
                grandchildren_ids.extend(self.childrenof(child_id, depth - 1))
            return grandchildren_ids


    def parentof(self, nodeid: int, depth: int = 1) -> Node:
        """
        Returns the parent of a given node, as identified by its node ID. If 'depth' is greater than 1, it returns 
        the grandparent (or further ancestors) instead. This method returns the entire parent node rather than just
        the parent's ID.
        
        Arguments:
            • nodeid: int; The ID of the node whose parent or ancestor is to be found.
            • depth: int; The degree of ascent from the node. Default is 1 (direct parent).
        
        Returns:
            • Node: The parent or ancestor of the node.
        
        Raises:
            ValueError: If 'nodeid' is not found in the list of node IDs.
        """
        if nodeid not in self.nodes: 
            raise ValueError(f"nodeid {nodeid} not in the list of node ids, called self.nlst!")
        if not isinstance(depth, int) or depth < 1: depth = 1
        parent_ = self.nodes[nodeid]
        while depth > 0:
            if parent_.parent < 0: return self
            parent_ = self.nodes[parent_.parent]
            depth -= 1
        return parent_


    def siblingsof(self, nodeid: int, depth: int = 1) -> list[int]:
        """
        Returns the siblings of a given node, as identified by its node ID. If 'depth' 
        is greater than 1, it returns the "cousins" (or further relatives) instead. 
        
        Arguments:
            • nodeid: int; The ID of the node whose siblings or relatives are to be found.
            • depth: int; The degree of relational distance from the node. Default is 1 (direct siblings).
        
        Returns:
            • list[int]: The IDs of the siblings or relatives of the node.
        
        Raises:
            ValueError: If 'depth' is less than 0.
        """
        if depth < 0: raise ValueError("Depth must be a non-negative integer.")
        if nodeid == 0: return [0]
        parent = self.parentof(nodeid, depth)
        return self.childrenof(parent.idnum, depth)


    def ancestorsof(self, nodeid: int, this_node_included: bool = False) -> list[int]:
        """
        Returns the list of ancestors from the root node to a given node, as identified by its node ID. 
        
        Arguments:
            • nodeid: int; The ID of the node whose ancestors are to be found.
            • this_node_included: bool; If True, includes the given node in the list of ancestors. Default is False.
        
        Returns:
            • list[int]: The IDs of the ancestors of the node.
        """
        node = self.nodes[nodeid]
        option_label = node.label
        ancestors, walking_nodeid = [], 0
        for letter in option_label:
            ancestors.append(walking_nodeid)
            walking_on_node = self.nodes[walking_nodeid]
            walking_nodeid = walking_on_node.options[abcs.index(letter)].idnum
        if this_node_included: ancestors += [nodeid]
        return ancestors


    def descendantsof(self, nodeid: int, descendants: list[int] = None) -> list[int]:
        """
        Returns all descendants of a given node, as identified by its node ID. 
        This includes children, grandchildren, and so on.
        
        Arguments:
            • nodeid: int; The ID of the node whose descendants are to be found.
            • descendants: list[int]; A list to which the descendants' IDs will 
                be added. Leave this argument as None.  Used for recursion.
        
        Returns:
            • list[int]: The IDs of the descendants of the node.
        """
        if descendants is None: descendants = []

        node = self.nodes[nodeid]
        for child in node.options:
            descendants.append(child.idnum)
            self.descendantsof(child.idnum, descendants)
        return sorted(descendants)


    def birth_order(self, nodeid: int) -> int:
        """
        Returns the birth order of a node, as identified by its node ID. The birth 
        order is the index of the node within the list of its siblings, where 0 
        represents the first-born, 1 represents the second-born, and so on.
        
        Arguments:
            • nodeid: int; The ID of the node whose birth order is to be found.
        
        Returns:
            • int: The birth order of the node.
        """
        return self.siblingsof(nodeid).index(nodeid)


    def sibling_groups(self) -> dict[int: list[list[tuple[int]]]]:
        """
        Returns groups of siblings for each level in the tree. For example, an output might look
        like [[(0,)], [(1, 2)], [(3, 4), (5, 6)]], where each sublist represents a level in the
        tree, and each tuple within that sublist represents a group of sibling nodes at that level.
        
        Returns:
            • list[list[tuple[int]]]: A nested list of tuples, where each tuple 
                represents a group of sibling nodes at a specific level in the tree.
        """
        sgroups = {level: [] for level in range(0, self.levels[-1]+1)}
        level, sgroups[0] = 1, [(0,)]
        while level < self.levels[-1] + 1:
            parents = [idx for idx, ppl in enumerate(self.levels) if ppl == level - 1]
            for par in parents:
                if not self.nodes[par].isleaf():
                    sgroups[level].append(tuple(self.childrenof(nodeid=par)))
            level += 1
        return sgroups


    def parent_groups(self, leafs: bool = False) -> dict[int: list[list[tuple[int]]]]:
        """
        Returns nodes at each level in the tree that are parents. This method can be used alongside 
        sibling_groups() to understand the tree's structure.  If 'leafs' is True, it returns leaf 
        nodes (those without children) at each level instead.
        
        Arguments:
            • leafs: bool; If True, returns leaf nodes at each level. Default is False (returns parent nodes).
        
        Returns:
            • dict: A dictionary where each key is a level in the tree and 
                the value is a list of parent (or leaf) nodes at that level.
        """
        pgroups, level = {level: [] for level in range(0, self.levels[-1] + 1)}, 0
        while level < self.levels[-1] + 1:
            parents = [idx for idx, ppl in enumerate(self.levels) if ppl == level]
            for par in parents:
                if self.nodes[par].isleaf() is leafs:
                    pgroups[level].append(par)
            level += 1
        return pgroups  


    def relative_ages(self, nodeid: int, age_relation: str | int = "==") -> list[int]:
        """
        Returns a list of node IDs based on their relative ages to a given node. The age relationship 
        is determined by 'age_relation', which can be a mathematical comparison operator (as a string), 
        a keyword describing the relationship, or an exact age difference (as an integer).
        
        Arguments:
            • nodeid: int; The ID of the node to which other nodes' ages are to be compared.
            • age_relation: str | int; The relationship between the ages of the given node 
                and the nodes to be found. Default is '=='.
        
        Returns:
            • list[int]: The IDs of the nodes whose ages satisfy the given relationship with the given node's age.
        
        Raises:
            ValueError: If 'age_relation' is not a supported comparison operator or keyword.
        """
        mylevel, age_nodes = self.nodes[nodeid].level, []

        "Handling integer or numerical string inputs."
        if isinstance(age_relation, str) and age_relation.isdigit(): 
            age_relation = int(age_relation)
        if isinstance(age_relation, int):
            for nodeidnum, ageofnode in enumerate(self.levels):
                if mylevel == ageofnode + age_relation: age_nodes.append(nodeidnum)
        else:
            "Converting phrases into supported mathematical relations."
            wordtorelations = {
                "peers of": "==", "elders of": ">", "minors of": "<", 
                "same age as": "==", "older than": ">", "younger than": "<", 
                "same age or older than": ">=", "same age or younger than": "<=", 
                "older or younger than": "!=", "younger or older than": "!="}
            if age_relation in wordtorelations: age_relation = wordtorelations[age_relation]

            supported_relations = ["==", "!=", "<=", ">=", "<", ">"]
            if age_relation not in supported_relations: 
                raise ValueError(f"age_relation {age_relation} not supported.  Supports: {supported_relations}!")
            
            "Appending node id numbers to age_nodes if node level satisfies age_relation."
            for nodeidnum, ageofnode in enumerate(self.levels):
                if eval(f"{mylevel} {age_relation} {ageofnode}"):
                    age_nodes.append(nodeidnum)

        return age_nodes
    

    def thisismy(self, mynodeid: int, theirnodeid: int, family_relation: None | str = None) -> str | bool:
        """
        Determines the familial or age relationship between two nodes, as identified by their node IDs. 
        If 'family_relation' is specified, it checks whether this is the correct relation and returns 
        True or False. If 'family_relation' is None, it returns the string describing the relationship.
        
        Arguments:
            • mynodeid: int; The ID of the first node.
            • theirnodeid: int; The ID of the second node.
            • family_relation: None | str; The relationship to check, or 
                None to return the relationship as a string. Default is None.
        
        Returns:
            • str | bool: If 'family_relation' is None, the relationship between the nodes as a
                string. Otherwise, a boolean indicating whether the given relationship is correct.
        
        Raises:
            ValueError: If 'family_relation' is not a supported relationship or None.
        """
        relation_types = ["self", "parent", "child", "sibling", "ancestor", "descendant", "peer", "elder", "minor"]
        if family_relation not in relation_types and family_relation is not None:
            raise ValueError(f"family relation must be one of the following: {relation_types}!")
        
        mylevel, theirlevel = self.nodes[mynodeid].level, self.nodes[theirnodeid].level
        
        if mylevel < theirlevel:
            relation = "descendant" if theirnodeid in self.descendantsof(nodeid=mynodeid) else "minor" 
            if relation == "descendant" and theirnodeid in self.childrenof(nodeid=mynodeid): relation = "child"
        elif mylevel > theirlevel:
            relation = "ancestor" if theirnodeid in self.ancestorsof(nodeid=mynodeid) else "elder" 
            if relation == "ancestor" and theirnodeid == self.nodes[mynodeid].parent: relation = "parent"
        else: relation = "self" if theirnodeid == mynodeid else "peer" if theirnodeid not in self.siblingsof(nodeid=mynodeid) else "sibling"

        if family_relation is None: return relation

        """Your child is also your descendant and also your minor but your minor is not always your child.  
        If relation is 'sibling' but family_relation is 'peer', this method will still return True even if 
        the family_relation was not specific enough because your sibling is also your peer."""
        general_relations = {
            "self":       ["self",       "sibling",    "peer"], 
            "parent":     ["parent",     "ancestor",   "elder"], 
            "child":      ["child",      "descendant", "minor"], 
            "sibling":    ["sibling",    "peer"], 
            "ancestor":   ["ancestor",   "elder"], 
            "descendant": ["descendant", "minor"], 
            "peer":       ["peer"], 
            "elder":      ["elder"], 
            "minor":      ["minor"]
            }
        relation = general_relations[relation]
        return family_relation in relation


    def nearest_common_ancestor(self, node_list: list[int]) -> int:
        """
        Returns the nearest common ancestor for a list of nodes.
        
        Each node in the tree is identified by a unique integer. This method traverses up the tree from 
        each node in the list and finds the nearest common ancestor (the node that is an ancestor of all 
        nodes in the list). If all nodes in the list are the same, it simply returns that node. If any 
        node is 0 (indicating the root node), it returns 0, as the root is an ancestor of all nodes. 
        It also implements some optimizations for common use cases to improve performance.
        
        Arguments:
            • node_list: list[int]; A list of integers representing node IDs in the tree.
            
        Returns:
            • int: The node ID of the nearest common ancestor.
            
        Raises:
            ValueError: If the provided node_list is not a list or is empty.
        """

        def _common_ancestor(node1: int, node2: int) -> int:
            """Finds the nearest common ancestor of node1 and node2."""

            "Optimizing for common use cases"
            if node1 == node2: return node1
            if any(0 == val for val in [node1, node2]): return 0
            if any(node1 == val for val in self.ancestorsof(node2)): return node1
            if any(node2 == val for val in self.ancestorsof(node1)): return node2
            if any(0 == val for val in [self.nodes[node1].parent, self.nodes[node2].parent]): return 0
            if any(node2 == val for val in self.siblingsof(node1)): return self.nodes[node1].parent

            temp_node1_id, temp_node2_id = node1, node2
            level1, level2 = self.nodes[temp_node1_id].level, self.nodes[temp_node2_id].level

            are_peers = level1 == level2
            while not are_peers:
                if level1 < level2:
                    temp_node2_id = self.parentof(temp_node2_id).idnum
                else: temp_node1_id = self.parentof(temp_node1_id).idnum
                level1, level2 = self.nodes[temp_node1_id].level, self.nodes[temp_node2_id].level
                are_peers = level1 == level2

            while not temp_node1_id == temp_node2_id:
                temp_node1_id = self.parentof(temp_node1_id).idnum
                temp_node2_id = self.parentof(temp_node2_id).idnum

            return temp_node1_id

        if not isinstance(node_list, (list, tuple)) or len(node_list) <= 0: 
            raise ValueError(f"node_list must be a list of integers, not {node_list}!")
        
        elif len(node_list) == 1: return node_list[0]

        nca = _common_ancestor(node1=node_list[-2], node2=node_list[-1])

        return self.nearest_common_ancestor(node_list=node_list[:-2] + [nca])


    def _epath(self, finish: int, start: int = 0) -> list[int]:
        """Produces an edge list, indicating the route from the start node to 
        the finish node, like [2, 6, 14, 20, 23, 28] if start = 2 and finish = 28. 
        This produces the same results as _opath()."""

        nca = self.nearest_common_ancestor(start, finish)
        fancestors = [ancestor for ancestor in self.ancestorsof(finish) if ancestor >= nca] + [finish]
        sancestors = [ancestor for ancestor in self.ancestorsof(start)  if ancestor >= nca] + [start]
        if nca == start: allancestors = fancestors
        elif nca == finish: allancestors = sancestors[::-1]
        else: allancestors = sancestors[::-1][:-1] + fancestors
        return allancestors        


    def _opath(self, finish: int, start: int = 0) -> list[int]:
        """Produces an edge list, indicating the route from the start node to 
        the finish node, like [2, 6, 14, 20, 23, 28] if start = 2 and finish = 28.  
        This relies on the alphabetical and hierarchical structure of option labels. 
        For example, an option label of BACA indicates option B, then option A, then 
        option C, then option A.  This is faster than _epath()."""

        commonality, opt = '', ''
        ostart = self.nodes[start].label
        ofinish = self.nodes[finish].label
        for os, of in zip(ostart, ofinish):
            if os == of: commonality += os
            else: break

        count, nca, nca_id = 0, self, 0
        if commonality != '': 
            while opt != commonality:
                count += 1
                opt = commonality[:count]
                nca = nca['options'][abcs.index(opt[-1])]
                nca_id = nca.idnum
        
        snca = fnca = nca
        sid = fid = int(nca.idnum)
        scount = fcount = len(commonality)
        sstr = fstr = str(commonality)
        sancestors, fancestors = [], [nca_id]
        while sid < start:
            sstr += ostart[scount]
            snca = snca['options'][abcs.index(sstr[-1])]
            sid = snca.idnum
            sancestors.append(sid)
            scount += 1
        while fid < finish:
            fstr += ofinish[fcount]
            fnca = fnca['options'][abcs.index(fstr[-1])]
            fid = fnca.idnum
            fancestors.append(fid)
            fcount += 1

        allancestors = sancestors[::-1] + fancestors
        return allancestors         
        

    def path(self, finish: int, start: int = 0) -> list[int]:
        """
        Calculates the path from a start node to a finish node within the game tree.
        
        This method takes two nodes as inputs, a start node and a finish node, and determines the 
        sequence of nodes one would traverse to get from the start node to the finish node within 
        the game tree. If there's a direct path from start to finish, the method uses the `_opath` 
        method. If not, the `_epath` method is used, which computes the path using an alternative 
        algorithm.
        
        Arguments:
            • start: int; A unique integer representing the start node ID in the tree.
            • finish: int; A unique integer representing the finish node ID in the tree.
        
        Returns:
            • list[int]: A list of integers representing the path of node IDs from the start node to 
            the finish node in the tree.
        """
        try: path = self._opath(finish, start)
        except: path = self._epath(finish, start)
        return path


    def distance(self, node1: int, node2: int, genetic_relatedness: bool = False) -> int | float:
        """
        Calculates the number of nodes between two given nodes in the tree.
        
        This method determines the path between two nodes, node1 and node2, and then returns the 
        length of this path. If genetic_relatedness is True, it returns the genetic relatedness 
        between the nodes, which is calculated as 0.5 raised to the power of the length of the path 
        between the nodes.
        
        Arguments:
            • node1: int; A unique integer representing a node ID in the tree.
            • node2: int; A unique integer representing another node ID in the tree.
            • genetic_relatedness: bool; If True, returns the genetic relatedness between node1 
            and node2. Defaults to False.
        
        Returns:
            • int | float: The number of nodes between node1 and node2, or the genetic relatedness 
            if genetic_relatedness is set to True.
        """
        path = self.path(finish=node1, start=node2)
        if genetic_relatedness: return .5**len(path)
        else: return len(path)


    """The following methods concern tree's timeline: timeline, timeline_is_paradoxical, and temporal_relation.
    The timeline is the path from the root node to the current node, as specified by current_nodeid."""

    def timeline(self) -> list[int]:
        """
        Finds the path from the root node to the current node in the tree.
        
        This method determines the sequence of nodes one would traverse from the root node 
        (node ID 0) to the current node, as specified by the attribute current_nodeid.
        
        Returns:
            • list[int]: A list of integers representing the path of node IDs from the root node to 
            the current node in the tree.
        """
        return self.path(finish=self.current_nodeid, start=0)


    def timeline_is_paradoxical(self) -> bool:
        """
        Checks if the timeline of the tree adheres to specific rules.
        
        The rules stipulate that the status of all nodes up to but not including the 'current_nodeid' should
        be 'after'; the 'current_nodeid' itself can be 'before', 'during', or 'after'; and all other nodes 
        should be 'before'. If these conditions are not met, it indicates that responses may have been 
        misplaced, 'current_nodeid' is incorrect, or both.
        
        NOTE this is not used on 'update_current_nodeid' but maybe it should be use to be even more careful.

        Returns:
            • bool: True if the timeline is paradoxical, otherwise False.
        """
        if self.current_nodeid == 0: return False
        if self.current_nodeid != self.timeline()[-1]: return True
        factual = self.timeline()
        counter_factual = list(set(self.nlst) - set(factual))
        factual.remove(int(self.current_nodeid))
        
        for node in factual: 
            if self.nodes[node].node_status() != 'after': return True
        for node in counter_factual: 
            if self.nodes[node].node_status() != 'before': return True
        return False


    def temporal_relation(self, compared_to: int, if_current_nodeid: int | None = None, symbolic: bool = True) -> tuple[bool, int] | str:
        """
        Computes the temporal relation between a node of interest and another node.
        
        Temporal relations include 'before', 'during', 'after' and their counterfactual versions. If 'symbolic' is True, 
        this method returns a tuple where the first element indicates if the nodes are in the same timeline (True) or
        alternative timelines (False), and the second element quantifies the temporal distance between the nodes. If
        'symbolic' is False, it returns a string describing the temporal relationship.
        
        Arguments:
            • compared_to: int; The node ID to compare with the current node.
            • if_current_nodeid: int | None; The node ID representing the "present". Default is the current node ID.
            • symbolic: bool; Determines the format of the returned temporal relation. Default is True.
            
        Returns:
            • tuple[bool, int] | str: The temporal relation in a symbolic or descriptive format.
        
        Raises:
            Exception: If 'compared_to' and 'if_current_nodeid' are not integer node IDs.
        """ 
        if if_current_nodeid is None: 
            if_current_nodeid = self.current_nodeid
        thenow, comnod = if_current_nodeid, compared_to
        if not all(isinstance(nid, int) for nid in [comnod, thenow]):
            raise Exception("compared_to and if_current_nodeid must both be integer node ids.")
        
        if thenow == comnod: return (True, 0) if symbolic else 'present'
        if self.thisismy(mynodeid=thenow, theirnodeid=comnod, family_relation="peer"):
            return (False, 0) if symbolic else 'alternate present'
        
        nowlev, reflev = self.nodes[thenow].level, self.nodes[comnod].level
        if nowlev > reflev: 
            if self.thisismy(mynodeid=thenow, theirnodeid=comnod, family_relation="ancestor"):
                return (True, reflev - nowlev) if symbolic else 'past'
            else: return (False, reflev - nowlev) if symbolic else 'alternate past'

        else: 
            if self.thisismy(mynodeid=thenow, theirnodeid=comnod, family_relation="descendant"):
                return (True, reflev - nowlev) if symbolic else 'possible future'
            else: return (False, reflev - nowlev) if symbolic else 'alternate future'


    """The following methods deal with information sets: _find_apexes_of_simultaneous_choice_pyramids, 
    _simultaneous_choices_are_valid, _simultaneous_choices_to_information_sets, and create_information_set."""

    def _find_apexes_of_simultaneous_choice_pyramids(self) -> list[bool]:
        """
        Determines the apexes of pyramid structures within game trees formed by simultaneous choices,
        these are locations that represent the start of simultaneous decision situations.
        
        Identifies apexes using these rules:
            • If a node is a simultaneous choice node and its parent is not, it's an apex.
            • If a node's parent is an apex, it is not an apex.
            • Otherwise, it's an apex.
        
        Returns:
            list[bool]: List of booleans representing whether each node is an apex (True) or not (False).
        """
        apexes = [None] * self.nlst[-1] + [None]
        while any(node is None for node in apexes):
            for nodeid in self.nlst:
                this_node: Node = self.nodes[nodeid]
                if this_node.choicetypeis("simultaneous"):
                    relevant_ancestors = [self.parentof(nodeid=nodeid, depth=levels_to_check\
                        ).idnum for levels_to_check in range(1, self.nplayers)]
                    
                    if nodeid == 0 or not self.parentof(\
                        nodeid=nodeid, depth=1).choicetypeis("simultaneous"):
                        apexes[nodeid] = True
                    else: 
                        for ancestor in relevant_ancestors:
                            if apexes[ancestor]: 
                                apexes[nodeid] = False  
                                break
                        if apexes[nodeid] is None:
                            apexes[nodeid] = True 

                else: apexes[nodeid] = False

        return apexes        


    def _simultaneous_choices_are_valid(self, stop_code_at_violations: bool = True) -> bool:
        """
        Validates the arrangement of simultaneous choice nodes in the game tree.
        
        • Checks the symmetry rule: If a parent is a simultaneous choice node, all its children 
        must be either simultaneous choice nodes or none can be simultaneous choice nodes.
        • Identifies apexes and ensures that all descendants 'n choosers - 1' levels down are simultaneous.
        
        Arguments:
            • stop_code_at_violations (bool): If True, raises an exception when a rule violation is found.

        Returns:
            bool: True if simultaneous choices are valid, otherwise False.
        """
        for nodeid in self.nlst:
            this_node: Node = self.nodes[nodeid]
            if this_node.choicetypeis("simultaneous"):
                if this_node.isleaf(): 
                    message = f"Violation at node {nodeid}: Node {nodeid} is a leaf node and yet is simultaneous!"
                    if stop_code_at_violations: raise Exception(message)
                    print(message)
                    return False
                children = self.childrenof(nodeid=nodeid, depth=1)
                simultaneous_children = [self.nodes[child].choicetypeis("simultaneous") for child in children if not self.nodes[child].isleaf()]
                all_same = all(simultaneous_children) or not any(simultaneous_children)
                if not all_same:
                    message = f"Violation at node {nodeid}: Children of simultaneous nodes must be all simultaneous or all sequential!"
                    if stop_code_at_violations: raise Exception(message)
                    print(message)
                    return False

        """Finding apexes: 
        if a node is a simultaneous choice node:
            if the parent of this node is not a simultaneous choice node, then this node must be an apex.  
            else:
                if the parent of this node is an apex, then this node is not an apex.
                else: this node is an apex.
        else: This node is not an apex."""
        apexes = self._find_apexes_of_simultaneous_choice_pyramids()

        """If a node is an apex, then all descendents n choosers - 1 levels down must be simultaneous."""
        for nodeid, isapex in enumerate(apexes):
            this_node: Node = self.nodes[nodeid]
            if isapex:
                nchoosers = sum(this_node.chooser)
                descendants = self.childrenof(nodeid=nodeid, depth=nchoosers-1)
                for descendant in descendants:
                    descendant = self.nodes[descendant]
                    if not descendant.choicetypeis(choicetype="simultaneous"):
                        message = f"Violation at node {nodeid}: Descendants of an apex must all be simultaneous!"
                        if stop_code_at_violations: raise Exception(message)
                        print(message)
                        return False
                    
        return True


    def _simultaneous_choices_to_information_sets(self) -> None:
        """
        Discovers information sets based on simultaneous choices and updates the info_set attributes of all nodes.
        
        Note: While information sets can exist without simultaneous choices, simultaneous choices 
        cannot exist without information sets. This method focuses on creating information sets stemming from simultaneous choices.
        
        • Validates simultaneous choices in the game tree.
        • Discovers apexes of the pyramidal information set structures.
        • If a node is an apex, sets all descendents 'n choosers - 1' levels down into the information set.
        """
        "Establish that simultaneous choices are placed on the appropriate nodes."
        self._simultaneous_choices_are_valid(stop_code_at_violations=True)

        "Discovering the apexes of the pyramidal information set structures."
        apexes = self._find_apexes_of_simultaneous_choice_pyramids()

        "If a node is an apex, then all descendents n choosers - 1 levels down are in the information set."
        for nodeid, isapex in enumerate(apexes):
            this_node: Node = self.nodes[nodeid]
            if isapex:
                info_set = [[nodeid]]
                this_node.info_set = info_set
                nchoosers = sum(this_node.chooser)
                for level in range(1, nchoosers):
                    descendants = self.childrenof(nodeid=nodeid, depth=level)
                    info_set.append(descendants)
                    for descendant in descendants:
                        descendant = self.nodes[descendant]
                        descendant.info_set = info_set


    def create_information_set(self, nodes_list: list[int]) -> None:
        """
        Creates an information set with the provided nodes and updates the info_set attributes of these nodes.
        Validates the conditions for the information set and throws an error if any game-theoretic principles are violated.
        
        Note: Assumes that if one child is in an information set, then all siblings are also part of this information set.
        
        Arguments:
            • nodes_list (list[int]): List of node ids to be included in the information set.
        
        Raises:
            ValueError: If the root node is included in the nodes_list.
            ValueError: If any leaf node is included in the nodes_list.
            ValueError: If nodes at different levels are included in the nodes_list.
            ValueError: If nodes with different choosers are included in the information set.
        """
        if len(nodes_list) == 1:
            """If nodes_list has only one nodeid, this the goal is probably to overwrite a pre-existing
            information set that placed multiple nodes within a simultaneous choice.  In this case, make 
            all nodes that used to be part of the same information set into solitary information sets."""
            overwrite_info_set = copy.deepcopy(self.nodes[nodes_list[0]].info_set)
            for level in overwrite_info_set:
                for nodeid in level:
                    self.nodes[nodeid].info_set = [[nodeid]]

        if 0 in nodes_list:
            raise ValueError("The root node cannot be part of an information set.")

        for nodeid in nodes_list:
            if self.nodes[nodeid].isleaf():
                raise ValueError(f"Node {nodeid} is a leaf node. Multiple leaf nodes cannot be in information sets!")

        levels_in_info_set = [self.nodes[nodeid].level for nodeid in nodes_list]
        if len(set(levels_in_info_set)) != 1:
            raise ValueError(f"Levels in information set must be consistent: {levels_in_info_set}!")

        apex = self.nearest_common_ancestor(node_list=nodes_list)
        info_set = [[apex]]

        depth = 1
        while True:
            children = self.childrenof(nodeid=apex, depth=depth)
            info_set.append(children)
            if any(child in children for child in nodes_list) \
                or depth > self.levels[-1]:
                break
            depth += 1

        for pyramid_layer in info_set:
            payoffs_in_layer  = [tuple(self.nodes[nodeid].payoffs) for nodeid in pyramid_layer]
            if len(set(payoffs_in_layer)) != 1:
                average_payoffs = [int(payoff) for payoff in np.average(payoffs_in_layer, axis=0)]
                for nodeid in nodes_list: self.nodes[nodeid].payoffs = average_payoffs
            choosers_in_layer  = [tuple(self.nodes[nodeid].chooser) for nodeid in pyramid_layer]
            if len(set(choosers_in_layer)) != 1:
                raise ValueError(f"Choosers in information set must be consistent: {choosers_in_layer}!")
            for nodeid in pyramid_layer:
                self.nodes[nodeid].info_set = info_set


    """The following three methods are used to assign attributes to trees: probabilites, choosers, and payoffs."""

    def assign_probabilities(self, probabilities: str | list[None | list[float, float]] \
            | dict[int: list[float, float]] = {}, print_template: bool = False) -> None:
        """
        Modifies the probabilities and ambiguity levels for each node.

        In a game tree, a node's probability determines the likelihood of transitioning to that node
        from its parent. The ambiguity level for a node represents the uncertainty of its probability.
        This function provides several ways to modify these probabilities and ambiguities.

        Arguments:
            • probabilities: Either a string command, list, or dict that represents the new probabilities.
                - string: Applies a global command.
                    • 'uniform': Equalizes probabilities of all siblings.
                    • 'deterministic': Makes probabilities 1 (selected) or 0 (not selected).
                    • 'nondeterministic': Assigns random probabilities.
                    • 'ambiguous': Sets all ambiguities to 0.5 (maximum uncertainty).
                    • 'unambiguous': Sets all ambiguities to 0.0 (no uncertainty).
                - list: List of probabilities and ambiguities for each node in the tree. 
                  A value of None implies no change to that node.
                  Example: [None, None, None, [.143, .2], [.286, .2], [.571, .2], None, None, None, None, [.67, .4], [.33, .4]]
                - dict: Specific probabilities and ambiguities for each node, where the key is the node ID and value is a 2-element list.
                  Example: {3: [.143, .2], 4: [.286, .2], 5: [.571, .2], 10: [.67, .4], 11: [.33, .4]}
            • print_template: If True, this will print an example dictionary of probabilities that
              developers can copy, paste, and change as needed. This saves time.

        Raises:
            ValueError: If probabilities argument is not recognized or has incorrect formatting.
        """
        if print_template:
            template = {nodeid: self.nodes[nodeid].probability for nodeid in self.nlst}
            pp.pprint(template), exit()
        else:
            if probabilities in ['uniform', 'nondeterministic', 'ambigious', 'unambigious']:
                for nodeID in self.nlst[1:]:
                    if probabilities ==   'uniform':          self.nodes[nodeID].probability = [1, self.nodes[nodeID].probability[1]]
                    elif probabilities == 'nondeterministic': self.nodes[nodeID].probability = [random.random(), self.nodes[nodeID].probability[1]]
                    elif probabilities == 'ambigious':        self.nodes[nodeID].probability = [self.nodes[nodeID].probability[0], 0.5]
                    elif probabilities == 'unambigious':      self.nodes[nodeID].probability = [self.nodes[nodeID].probability[0], 0.0]

            elif probabilities == 'deterministic':
                siblings_lst = []
                for peers in self.sibling_groups().values():
                    for siblings in peers:
                        siblings = list(siblings)
                        random.shuffle(siblings)
                        siblings_lst.append(siblings)          

                for siblings in siblings_lst[1:]:
                    for idx, child in enumerate(siblings):
                        if idx == 0: self.nodes[child].probability = [1, 0]
                        else: self.nodes[child].probability = [0, 0]

            elif isinstance(probabilities, str): 
                raise ValueError("Supported strings: 'uniform', 'deterministic', 'nondeterministic', 'ambigious', 'unambigious'")

            elif isinstance(probabilities, list):
                probabilities = {nodeID: probabilities[nodeID] for nodeID \
                    in self.nlst if probabilities[nodeID] is not None}

            if isinstance(probabilities, dict):
                "If user forgets to enter ambiguities, this sets them to 0 by default."
                probabilities = {nodeID: [probabilities[nodeID], 0] if isinstance(probabilities[\
                    nodeID], (int, float)) else probabilities[nodeID] for nodeID in probabilities.keys()}

                "If user enters probabilities for only a subset of siblings, this adjusts the sibling probabilities."
                for nodeID in list(probabilities.keys()):
                    if isinstance(nodeID, str):
                        try: nodeID = int(nodeID)
                        except: raise ValueError("Node IDs in the probabilities must be integers or strings of integers.")
                    current_node = self.nodes[nodeID]
                    siblings_lst = self.siblingsof(nodeid=current_node.idnum)
                    sibling_probs = [probabilities[node][0] if node in probabilities.keys() else None for node in siblings_lst]
                    n_unspecified  =     len([prob for prob in sibling_probs if prob is None])
                    remaining_prob = 1 - sum([prob for prob in sibling_probs if prob is not None])
                    if n_unspecified > 0:
                        unspecified_prob = round(remaining_prob / n_unspecified, 4)
                        for idx, sibling in enumerate(siblings_lst):
                            if sibling != nodeID and sibling_probs[idx] is None: 
                                self.nodes[sibling].probability = [unspecified_prob, 0]

                    current_node.probability = probabilities[nodeID]

            def norm_probs(self):
                if not self.isleaf():
                    children_prob_sum = sum([child.probability[0] for child in self.options])
                    for child in self.options:
                        child_prob = list(child.probability)
                        if child_prob[1] > 1: child_prob[1] = 1
                        child_prob[0] = round(child.probability[0] / children_prob_sum, 3)
                        child.probability = child_prob
                        norm_probs(child)

            norm_probs(self)
            self.probability = [1, 0]


    def assign_choosers(self, choosers: dict[int:list[bool]] = {}, print_template: bool = False) -> None:
        """
        Assigns choosers to nodes by node ids.

        Choosers determine who makes decisions at each node in the game tree. Each 
        player's role at each node is represented as a Boolean value in a list.
        
        Arguments:
            • choosers: Dictionary where the key is the node ID and value is a
                list of Booleans representing the players' roles at that node.
                example_choosers = {
                    0: [True, False, False], 
                    1: [False, True, False], 
                    2: [False, False, True], 
                    3: [False, False, False],... 
                    11: [False, True, True]
                    }
                In this example, node 0 is chosen by the first player, node 1 by 
                the second, node 2 by the third, node 3 by chance, and node 11 by 
                the second and third players simultaneously.  Note that all nodes 
                are chance nodes by default.
            • print_template: If True, prints a template of 'choosers' for your 
                tree that you can fill out instead of typing everything manually.

        Raises:
            ValueError: If the choosers argument is not formatted correctly.
        """
        if print_template:
            choosers = {nodeid: [False]*self.nplayers for nodeid in self.nlst if not self.nodes[nodeid].isleaf()}
            pp.pprint(choosers), exit()
        else:
            "Setting random sequential choice chooser if choosers == 'random'." 
            if choosers == "random":
                choosers = {nodeid: [False]*self.nplayers for nodeid in self.nlst if not self.nodes[nodeid].isleaf()}
                for nodeid in list(choosers.keys()):
                    this_chooser = choosers[nodeid]
                    achooser = random.randint(0, self.nplayers - 1)
                    this_chooser[achooser] = True

            "Setting the first chooser to update self.nplayers."
            first_node, first_chooser = list(choosers.items())[0]
            chooser_length = len(first_chooser)
            for chooser_ in choosers.values():
                if len(chooser_) != chooser_length:
                    raise ValueError("All choosers are lists that must have the same length.")
            self.nodes[first_node].chooser = first_chooser

            "Setting chooser attributes"
            for nodeid, chooser_ in choosers.items():
                if self.nodes[nodeid].isleaf(): raise ValueError("Leaf nodes cannot be assigned choosers!")
                if len(chooser_) != self.nplayers or any([not isinstance(chooser, bool) for chooser in chooser_]):
                    raise ValueError(f"Entries must be lists of bools with a length equal to the number of players, like {[False]*self.nplayers}")
                self.nodes[nodeid].chooser = chooser_
            self._simultaneous_choices_to_information_sets()


    def assign_payoffs(self, payoffs: dict[int:list[bool]] = {}, print_template: bool = False) -> None:
        """Assign payoffs to nodes by node ids.  Works like assign_choosers().

        Arguments:
            • payoffs: Dictionary where the key is the node ID and value is a 
                list of integers representing the players' payoffs at that node.
                example_payoffs = {
                    0:  [8, 3, 4],
                    1:  [7, 9, 4],
                    2:  [8, 3, 6],
                    3:  [9, 5, 0],
                    5:  [7, 0, 2],
                    11: [6, 1, 5]
                }
            • print_template: If True, prints a template of 'payoffs' for your 
                tree that you can fill out instead of typing everything manually.         

        Raises:
            ValueError: If payoff lists are of unequal length
            ValueError: If lengths payoff list(s) do not match 
                the number of players in the tree         
        """

        if print_template:
            payoffs = {nodeid: [0]*self.nplayers for nodeid in self.nlst}
            pp.pprint(payoffs), exit()
        else:
            "Setting random payoffs if payoffs == 'random'." 
            if payoffs == "random":  
                payoffs = {nodeid: [0]*self.nplayers for nodeid in self.nlst}
                for nodeid in list(payoffs.keys()):
                    this_payoff = payoffs[nodeid]
                    for payoff in range(self.nplayers):
                        this_payoff[payoff] = random.randint(1, 9)              
           
            "Setting the first payoff to update self.nplayers."
            first_node, first_payoff = list(payoffs.items())[0]
            payoff_length = len(first_payoff)
            for payoff_ in payoffs.values():
                if len(payoff_) != payoff_length:
                    raise ValueError("All payoffs are lists that must have the same length.")
            self.nodes[first_node].payoffs = first_payoff
                
            "Setting payoff attributes"    
            for nodeid, payoff_ in payoffs.items():
                if len(payoff_) != self.nplayers or any([not isinstance(payoff, (int, float)) for payoff in payoff_]):
                    raise ValueError(f"Entries must be lists of numbers with a length equal to the number of players, like {[0]*self.nplayers}")
                self.nodes[nodeid].payoffs = payoff_


    def randomize_payoffs(self, min_payoff: int, max_payoff: int, step_size: int | float = 1, 
                          only_apply_payoffs_to_leaf_nodes: bool = True) -> None:
        """
        Assigns payoffs randomly to the tree.

        Arguments:
            • min_payoff: int; The minimum payoff
            • max_payoff: int; The maximum payoff
            • step_size: int | float; The intervals between the min_payoff and 
                max_payoff used to create an array of possible payoff values. 
            • only_apply_payoffs_to_leaf_nodes: bool (True); 
                - If True, payoffs are only applied to leaf nodes
                - If False, payoffs are applied to all nodes      
        """
        "Create list of payoffs to randomly sample from"
        payoff_array = list(range(min_payoff, max_payoff + 1, step_size))

        "Iterate over leaf nodes"
        for nodeid in self.nlst:
            this_node: Node = self.nodes[nodeid]
            "Exclude parent nodes if only_apply_payoffs_to_leaf_nodes"
            if this_node.isleaf() or not only_apply_payoffs_to_leaf_nodes:
                "Apply randomly selected payoffs"
                for payoff_idx in range(len(this_node.payoffs)):
                    this_node.payoffs[payoff_idx] = random.choice(payoff_array)


    def node_coordinates(self, screen_width: float = 1.0, screen_height: float = 1.0, 
            round_digits: int | None = 6, randomize_visual_permutation: bool = True) -> dict[int: list[float, float]]:
        """
        Determines optimal (x, y) coordinates for each node to produce a visually appealing tree layout. 
        The coordinates are applied to the nodes in the 'positionxy' attribute. Inspired by Joel's answer 
        at https://stackoverflow.com/a/29597209/2966723.

        This method organizes nodes with even spacing along the y-axis, and recursively distributes 
        children nodes along the x-axis. Finally, the method provides rounding for aesthetic purposes, 
        which increases precision with each level in the tree.

        Arguments:
            • screen_width/height: float; Dimensions for resizing the tree to fit your screen.
            • round_digits: int | None; Number of digits for rounding the coordinates (optional).
            • randomize_visual_permutation: bool; If True, it randomizes the tree's visual arrangement 
                to control for the effect of positioning.

        Returns:
            • dict: A dictionary mapping node IDs to their (x, y) coordinates.
        """
        "It is pointless to run this on tiny trees."
        if len(list(self.nodes.keys())) <= 1: return {0: [0.5, 1.0]}
        elif len(list(self.nodes.keys())) == 2: return {0: [0.5, 1.0], 1: [0.5, 0.0]}
        elif len(list(self.nodes.keys())) == 3: return {0: [0.5, 1.0], 1: [0.0, 0.0], 2: [1.0, 0.0]}

        "Starting parent position"
        root_position = [0.5, 1.0]

        "Establishing a constant spacing between levels."
        max_level = self.levels[-1]
        yaxis_gap = 1.0 / max_level

        def node_coords(parentid: int = 0, parent_position = root_position, xaxis_gap: float = 0.5, 
                yaxis_gap: float = yaxis_gap, position_dict: dict[int: [float, float]] = None):
            """Parents provide their children an xaxis_gap, which is the space they own.  Siblings 
            must divide this space equally, as seen in xaxis_gap_ = xaxis_gap / len(children).  When
            each sibling becomes a parent, they provide a smaller sliver to their children, who will
            provide a smaller sliver to their children, and so on..."""

            if position_dict is None: position_dict = {parentid: parent_position}
            else: position_dict[parentid] = parent_position 
            children = self.childrenof(nodeid=parentid, depth=1)
            if not self.nodes[parentid].isleaf():
                xaxis_gap_ = xaxis_gap / len(children)
                leftmost_childx = parent_position[0] - xaxis_gap / 2 - xaxis_gap_ / 2
                options_idxs = list(range(len(self.nodes[parentid].options)))
                if randomize_visual_permutation: 
                    random.shuffle(options_idxs)
                for idx in options_idxs:
                    child = self.nodes[parentid].options[idx]
                    leftmost_childx += xaxis_gap_
                    position_dict = node_coords(parentid=child.idnum, parent_position=[leftmost_childx, parent_position[1] \
                        - yaxis_gap], xaxis_gap=xaxis_gap_, yaxis_gap=yaxis_gap, position_dict=position_dict)
            return position_dict
        
        position_dict = node_coords(parentid = 0, parent_position = root_position, 
            xaxis_gap = 0.5, yaxis_gap = yaxis_gap, position_dict = None)
        
        "Creating 'permutation' attribute that expresses how the nodes are arranged."
        self.visual_permutation()

        "Rounding coordinates for aesthetic purposes"
        if round_digits is None: round_digits = 20

        for nodeid in position_dict:
            "Rounding digits must grow as the levels increase because lower nodes need more precise coordinates."
            this_node_level = self.nodes[nodeid].level
            roundto = round_digits + this_node_level - 6 if this_node_level > 6 else round_digits
            position_dict[nodeid][0] = round(position_dict[nodeid][0] * screen_width,  roundto)
            position_dict[nodeid][1] = round(position_dict[nodeid][1] * screen_height, roundto)        

        "Sorting the position dictionary by integer keys."
        sorted_keys = sorted(position_dict.keys())
        sorted_position_dict = {}
        for key in sorted_keys:
            sorted_position_dict[key] = position_dict[key]

        return sorted_position_dict


    def seconds_on_nodes(self, seconds_per_node: int = 12, round_start_time: int = 0, 
        seconds_per_descendant: int = 1, buffer_between_levels: int = 0, print_: bool = False) -> None:
        """
        Calculates and assigns the start and end times for each node in a game tree.

        The method determines the time duration allocated for each node considering the number of descendants 
        each node has. This duration is added to a start time to determine a window (start_time, end_time) 
        during which players can respond. A buffer time is also added before the next node's start time.

        Arguments:
            • seconds_per_node: int; The base duration in seconds each node is given. Defaults to 12.
            • round_start_time: int; The starting timestamp of the round. If not an integer or float, 
                the current timestamp is used. Defaults to 0.
            • seconds_per_descendant: int; Additional time awarded based on the number of descendants.
                - Adds 1.5x time to the root node because people want to think the most at the root.
            • buffer_between_levels: int; Grace period added to the start time of the next node. 
                This provides backend processing time during level transitions. Defaults to 0.
            • print_: bool; If set to True, the start and end times are printed. Defaults to False.

        This method modifies the 'time' attribute of each node in the tree, which is a tuple of (start_time, end_time).
        """
        if not isinstance(round_start_time, (int, float)): 
            round_start_time = datetime.datetime.now().timestamp()
        
        def putseconds(tree, seconds_per_node, node_start_time, seconds_per_descendant, buffer_between_levels):
            bonus_for_descendantsof = seconds_per_descendant * len(self.descendantsof(nodeid=tree.idnum))
            if tree.idnum == 0: bonus_for_descendantsof = int(bonus_for_descendantsof * 1.5) 
            tree.time = (node_start_time, node_start_time + seconds_per_node + bonus_for_descendantsof)
            if print_: print(f"Node {tree.idnum}: {tree.time[0]} to {tree.time[1]}")
            for child in tree.options:
                putseconds(tree=child, seconds_per_node=seconds_per_node, 
                    node_start_time=tree.time[1] + buffer_between_levels, 
                    seconds_per_descendant=seconds_per_descendant, 
                    buffer_between_levels=buffer_between_levels)

        putseconds(self, seconds_per_node, round_start_time, seconds_per_descendant, buffer_between_levels)


    def duration(self, question_type: str = "max") -> float:
        """
        Estimates the duration to play the tree based on different types of questions such as 'max', 
        'min' or 'avg'. The question determines whether we're seeking the longest possible, shortest 
        possible or average duration of play on the tree.

        Arguments:
            • question_type: str; Determines the type of duration calculation.
                - 'max': Returns the maximum possible duration.
                - 'min': Returns the minimum possible duration.
                - 'avg': Returns the average possible duration.

        Returns:
            • float: The calculated duration.

        Raises:
            ValueError: If the input for 'question_type' is not 'max', 'min' or 'avg'.
        """
        round_start = self.time[0]

        if question_type == "max":
            max_level = max(self.levels)
            for node, lev in enumerate(self.levels):
                if lev >= max_level: 
                    return round(self.nodes[node].time[1] - round_start, 3)
        elif question_type == "min":
            ending_times = []
            for node in self.nlst:
                if self.nodes[node].isleaf(): 
                    ending_times.append(round(self.nodes[node].time[1] - round_start, 3))
                    return min(ending_times)
        elif question_type == "avg":
            duration_sum, n_leaves = 0, 0
            for node in self.nlst:
                if self.nodes[node].isleaf():
                    duration_sum += self.nodes[node].time[1]
                    n_leaves += 1
            return round(duration_sum / n_leaves  - round_start, 3)
        else: raise ValueError("Invalid Input: question_type must be 'max' or 'avg'!")


    @classmethod
    def durations(self, list_of_trees: list['Tree'], question_type: str = "max"):
        """
        Calculates the expected play duration for a round of the experiment. This class method 
        uses the 'duration' method to estimate the time for each tree and then calculates the 
        desired statistic (max, min, or average) across all trees.  

        Arguments:
            • list_of_trees: list; A list of Tree objects for which to calculate the duration.
            • question_type: str; Determines the type of duration calculation for the list.
                - 'max': Returns the maximum possible duration.
                - 'min': Returns the minimum possible duration.
                - 'avg': Returns the average possible duration.

        Returns:
            • float: The calculated duration statistic.

        Raises:
            ValueError: If the input for 'question_type' is not 'max', 'min' or 'avg'.        
        """
        supported_question_types = ["max", "min", "avg"]
        if question_type not in supported_question_types:
            raise ValueError(f"question_type must be in {supported_question_types}!")

        durations_ = []
        for tree in list_of_trees:
            durations_.append(tree.duration(question_type=question_type))

        if question_type == "max": return max(durations_)
        elif question_type == "min": return min(durations_)
        else: return np.average(durations_)


    def _max_reaction_time(self, nodeid: int = None, subtract_from_time_limit: bool = False) -> float | int:
        """
        Determines the longest reaction time stored within responses on a node.
        This method helps calculate the remaining time in a node, so that this
        time can be subtracted from all descendants.

        Arguments:
            • nodeid: int (optional); The id number for the 
                node in question.  Defaults to the current node.
            • subtract_from_time_limit: bool (optional); If True, 
                will subtract from time limit.  Defaults False.

        Returns:
            • float | int: The maximum reaction time at a node.
        """
        if nodeid is None: nodeid = self.current_nodeid
        this_node = self.nodes[nodeid]
        start_time, end_time = this_node.time[0], this_node.time[1]
        if this_node.isleaf() or this_node.node_status() != "after": 
            return end_time - start_time if subtract_from_time_limit else 0.0

        maxrt = 0.0
        for chooser, choice, prediction in zip(this_node.chooser, this_node.choice, this_node.prediction):
            "If the response reaction time is greater than maxrt, maxrt becomes this response reaction time."
            if chooser: 
                if choice['rtimedn'] > maxrt: 
                    maxrt = choice['rtimedn']
            else: 
                "If a player is not a chooser, then they are a predictor."
                if not this_node.choicetypeis("chance"):
                    if prediction['rtimedn'] > maxrt: maxrt = prediction['rtimedn']
  
        return round(end_time - maxrt if subtract_from_time_limit else maxrt, 6)
        

    # def apply_earnings(self) -> None:
    #     """
    #     Apply cummulative payoffs and clear memoization cache.
    #     """
    #     if self.experiment is not None:
    #         if self.nodes[self.current_nodeid].isleaf():

    #             cumulative_payoffs = self.earnings()
    #             for player, cum_payoff in zip(self.players, cumulative_payoffs):
    #                 player['cumulative_payoffs'] += cum_payoff

    #             "Delete memo is one exists"
    #             if 'memo' in self: del self['memo']


    def update_delay(self) -> float:
        """
        Calculates how many more seconds until the current node id should be updated on the frontend.

        Delays are necessary to allow human participants to see responses from chance nodes and artificial agents.

        This must be called after the current node id has already been updated.
        """
        if self.current_nodeid == 0: return 0.0
        current_node_parent = self.nodes[self.nodes[self.current_nodeid].parent]
        delay, current_time = 0.0, time.time()

        if self.choicetypeis("chance"):
            for idx in range(self.nplayers):
                response = current_node_parent.choice[idx]
                if response is not None:
                    new_delay = response['timestamp'] + response['rtimeup'] - current_time
                    if new_delay > delay: return new_delay

        else:
            for idx in range(self.nplayers):
                if self.players[idx]['player_type'] == 'robot':
                    response = None
                    if current_node_parent.choice[idx] is not None:
                        response = current_node_parent.choice[idx]
                    elif current_node_parent.prediction[idx] is not None:
                        response = current_node_parent.prediction[idx]     
                    if response is not None: 
                        new_delay = response['timestamp'] + response['rtimeup'] - current_time         
                        if new_delay > delay: delay = new_delay      

        return delay


    def update_sids(self) -> None:
        """
        Updates websocket ids for all human players.

        NOTE: Assumes that the tree is part of an ongoing experiment.
        """
        if self.experiment is not None and self.experiment.experiment_manager is not None:
            "If the tree is part of an ongoing experiment..."

            for player in self.players:
                if player["player_type"] != "robot":
                    "...then get sids stored in the experiment manager for all human players."
                    player_sid = self.experiment.experiment_manager.uuids_to_sids.get(player['uuid'], None)
                    if player_sid is not None: player["sid"] = player_sid


    def emit_to_players(self, player_sid: str = None, update_timestamps: bool = True) -> None:
        """
        Emits game tree as serialized JSON object to all human players stored at the root.

        Arguments:
            • player_sid: str (default None); Websocket id of the player this will emit to.
                - Only enter an sid here if you want to emit just to one player, such as after 
                    this player refreshes the page.  Otherwise, this emits to all human players. 
            • update_timestamps: bool (default True); 
                - If the user refreshes their page mid-round, set to False
                - At the start of the round, set to True

        Message Type: 'game_tree'

        Payload: JSON serialized tree
        """
        if self.exeriment is None or (self.exeriment is not None and self.experiment.status == "active"):
            "Only emit if the experiment is active."
            if [player['player_type'] != 'robot' for player in self.players]:
                "If there is at least one human, then begin emit process."

                "Frontend uses timestamps to syncronize clock and update current node."
                current_time = time.time()               
                self.timestamps['emit_tree_time'] = current_time    
                if update_timestamps:  
                    remaining_time = self.time[1] - self.time[0] + current_time
                    self.timestamps['update_node_time'] = current_time + self.update_delay()     
                    self.timestamps['abdicate_node_time'] = remaining_time   

                "Make adjacency matrix JSON serializable."
                self.adjacency_matrix = list(self.adjacency_matrix)

                "Ensure websocket ids are current."
                self.update_sids()

                if isinstance(player_sid, str):
                    emit('game_tree', self, to=player_sid)

                else:
                    for player in self.players:
                        if player["player_type"] != "robot":
                            emit('game_tree', self, to=player["sid"])


    def emit_updated_tree(self) -> None:
        """
        Emits updated Tree to frontend players.

        Message Type: 'game_tree'

        Payload: JSON serialized tree
        """
        current_node = self.nodes[self.current_nodeid]
        error_message_end = f"to the current node {self.current_nodeid} before the tree was emitted!"
        for idx, choice_loc in enumerate(current_node.choice):
            if choice_loc is not None:
                raise Exception(f"Player {idx} submitted a choice to " + error_message_end)
        for idx, prediction_loc in enumerate(current_node.prediction):
            if prediction_loc is not None:
                raise Exception(f"Player {idx} submitted a prediction to " + error_message_end)

        self.emit_to_players()


    def play_game(self) -> None:
        """
        Initiates game for artificial agents.  If all players 
        are artificial, they will play the game to completion.

        Raises:
            Exception: If called on terminal node.
        """
        if self.isleaf():
            raise Exception("play_game() cannot run on terminal nodes.")
        if self.choicetypeis("chance"):
            self.choice_by_chance_node()
        else:
            for idx, player in enumerate(self.players):
                if player["player_type"] == "robot":
                    ag.agent(tree=self, player_tag=idx)


    def chosen_child(self, parent_: int | Node) -> Node:
        """
        Returns the chosen child node if all decisions have been made at the parent node.
        If not, returns the parent node.  Invoked by 'update_current_nodeid' to determine
        which node the game should progress to.

        Arguments:
            • parent_ (int | Node): An integer (ID of the parent node) or an instance of the Node class.

        Returns:
            • Node: The chosen child node if all decisions have been made at the parent node, otherwise the parent node.

        Raises:
            Exception: If the parent node is of type "chance" or "sequential" but no choices have been made yet.
            ValueError: If the parent node is of type "simultaneous" but does not correspond to the apex node.
        """

        if isinstance(parent_, int): parent_ = self.nodes[parent_]

        violation_message = f"No choices applied to node {parent_.idnum}!"

        if parent_.isleaf() or not parent_.finished_choosing(): return parent_

        if parent_.choicetypeis(choicetype="chance"):
            for choice in parent_.choice:
                if choice is not None:
                    child_labels = [child.label for child in parent_.options]
                    child_index = child_labels.index(choice["option"])
                    return parent_.options[child_index]
            raise Exception(violation_message)
        
        elif parent_.choicetypeis(choicetype="sequential"):
            for chooser, choice in zip(parent_.chooser, parent_.choice):
                if chooser: 
                    child_labels = [child.label for child in parent_.options]
                    child_index = child_labels.index(choice["option"])
                    return parent_.options[child_index]
            raise Exception(violation_message)
        
        elif parent_.choicetypeis(choicetype="simultaneous"):
            apex_node = parent_.info_set[0][0]

            if parent_.idnum != apex_node:
                raise ValueError(f"Parent id number {parent_.idnum} ≠ apex node id {apex_node}!")

            choice_labels = [choice["option"] for choice in parent_.choice if choice is not None]

            choice_label = str(parent_.label)
            for label in choice_labels:
                choice_label += label[-1]

            return self.nodes[self.olst.index(choice_label)]		


    def update_current_nodeid(self) -> None:
        """
        Updates the current node ID after ensuring that all responses have been applied to the current node.
        This function also checks for temporal paradox errors, and emits the tree to the frontend players.

        Raises:
            AssertionError: If the status of the node is not "after" when it should be.

        Side-effects:
            Updates 'current_nodeid' property of the tree and may adjust the remaining time for descendant nodes.
            Emits the updated tree to the frontend.
        """
        if self.tree_status != "after":
            now_node = self.nodes[self.current_nodeid]

            if now_node.node_status() == "after":
                chosen_child_idnum = self.chosen_child(parent_=self.current_nodeid).idnum 

                if chosen_child_idnum != now_node.idnum:
                    "Updating current_nodeid to the chosen child idnum."
                    self.current_nodeid = chosen_child_idnum

                    mypeers = self.relative_ages(nodeid=now_node.idnum, age_relation="peers of")
                    status_of_peers = [self.nodes[peer].node_status() for peer in mypeers]

                    for peer, status in zip(mypeers, status_of_peers):
                        "Eliminating temporal paradoxes."
                        if peer != now_node.idnum and status != "before": 
                            self.delete_responses(nodeid=peer)

                        "Making responses at this level permanent by converting to immutable type."
                        self.nodes[peer].choice = tuple(self.nodes[peer].choice)
                        self.nodes[peer].prediction = tuple(self.nodes[peer].prediction)

                    new_node = self.nodes[self.current_nodeid]
                    new_node.incriment_cumulative_payoffs()

                    "Subtracting remaining time from descendants."
                    if not new_node.isleaf():
                        maximum_reaction_time = self._max_reaction_time(nodeid=now_node.idnum, subtract_from_time_limit=False)
                        remaining_time = int(now_node.time[1] - maximum_reaction_time)
                        descendants = self.descendantsof(nodeid=now_node.idnum)
                        for descendant in descendants:
                            descendant = self.nodes[descendant]
                            descendant.time = (descendant.time[0] - remaining_time, descendant.time[1] - remaining_time)

                    "Updating the round_room_batch attribute to reset the frontend countdown timer"
                    round_duration = new_node.time[1] - new_node.time[0]
                    self.round_room_batch = (self.round_room_batch[0], self.round_room_batch[1], self.round_room_batch[2], int(time.time() + round_duration))

                    avatars = [player['avatar']['shape'] for player in self.players]
                    print(f"Updating current node {now_node.idnum} to {self.current_nodeid}. emit {self.title} to players: {avatars}")
                    self.emit_updated_tree()

                    if not new_node.isleaf():
                        if new_node.choicetypeis("chance"):
                            new_node._root.choice_by_chance_node()
                        else:
                            "Telling artificial agents to submit responses"
                            for idx, player in enumerate(self.players):
                                if player["player_type"] == "robot":
                                    ag.agent(tree=self, player_tag=idx)


    def delete_responses(self, nodeid: int | None = None) -> None:
        """
        Deletes all respones from a specific node or all nodes if nodeid is None.
        """
        for node in self.nodes.values():
            if node.idnum == nodeid or nodeid is None:
                ischance = self.choicetypeis(choicetype="chance")
                for playernum in range(self.nplayers):
                    if node.chooser[playernum] or ischance:
                        choice = node.choice[playernum]
                        if choice is not None:
                            print(f"At node {node.idnum} for player {self.players[playernum]}, deleted choice {choice}.")
                            node.choice[playernum] = None
                    if node.predictor[playernum] or ischance:
                        prediction = node.prediction[playernum]
                        if prediction is not None:
                            print(f"At node {node.idnum} for player {self.players[playernum]}, deleted prediction {prediction}.")
                            node.prediction[playernum] = None


    def _randomly_selected_option(self, parent_nodeid: int = None, player_num: int = None) -> str:
        """
        Selects a child node of the parent node based on the 'probability' attributes of the children.

        This method is used to make selections for chance nodes and for abdicated responses.

        Arguments:
            • parent_node: int; The id number of the parent node.
            • player_num: int; This is optional unless the parent node is a simultaneous choice
                node.  In this case, the player number determines the length of the node label

        Returns:
            • str: The label of the selected child node.
        
        Raises:
            • TypeError: If parent_nodeid is not an int or None.
            • Exception: If the parent node is a leaf node.    
            • TypeError: If player_num is not an int or None.
            • ValueError: If player_num not in tree.  
        """
        parent_nodeid = self.current_nodeid if parent_nodeid is None else \
            parent_nodeid.idnum if isinstance(parent_nodeid, Node) else parent_nodeid
        if not isinstance(parent_nodeid, int):
            raise TypeError(f"Unsupported type for parent_nodeid {type(parent_nodeid)}.")
        this_node = self.nodes[parent_nodeid]

        if this_node.isleaf():
            raise Exception(f"_randomly_selected_option called on leaf node {parent_nodeid}!")

        if this_node.choicetypeis("simultaneous"):
            if not isinstance(player_num, int):
                raise TypeError(f"At simultaneous choice nodes, player_num must be given as an integer, not {type(parent_nodeid)}.")
            if not (0 <= player_num < self.nplayers):
                raise ValueError(f"player_num {player_num} not found in {self.nplayers}-player tree!")

            player_role = "chooser" if this_node.chooser[player_num] else "predictor"

            if player_role == "chooser":
                chooser_num = sum([int(chooser) for chooser in this_node.chooser[:player_num]])
                parent_nodeid = random.choice(this_node.info_set[chooser_num])
            else: parent_nodeid = random.choice(this_node.info_set[-1])

        children = self.childrenof(parent_nodeid)
        child_labels = [self.olst[child] for child in children]

        child_probs = [self.nodes[self.olst.index(child)].draw_probability() for child in child_labels]
  
        return random.choices(child_labels, weights=child_probs, k=1)[0] 
        

    def choice_by_chance_node(self) -> None:
        """
        Applies a random choice to the current node, based on the probabilities of the children.

        Raises:
            • ValueError: If the current node is a leaf node or not a chance node.

        Side-effects:
            • Modifies the 'choice' property of the current node.
            • Adds a two second reaction time so the frontend can delay displaying the choice.
        """
        current_node = self.nodes[self.current_nodeid]

        if current_node.isleaf(): 
            raise ValueError(f"Node id: {self.current_nodeid} is a leaf node!")

        if not current_node.choicetypeis("chance"): 
            raise ValueError(f"Node {self.current_nodeid} is not a chance node!")
                  
        natures_choice = {'option': self._randomly_selected_option(parent_nodeid=current_node, 
                          player_num=None), 'keypress': None, 'rtimedn': 2.0, 'rtimeup': 2.1, 'timestamp': time.time()}  

        current_node.choice = [natures_choice for playernum in range(self.nplayers)]


    def someone_abdicated(self, seconds_since_round_started: float | int, nodeid: int = None) -> bool:
        """
        Checks if any player failed to respond within the allotted time.  Used by 'response_to_tree'.

        Arguments:
            • seconds_since_round_started (float | int): The number of seconds that have passed since the round started.
            • nodeid (int, optional): The ID of the node to check. Defaults to the current node.

        Returns:
            • bool: True if any player failed to respond within the allotted time, False otherwise.

        Raises:
            ValueError: If 'seconds_since_round_started' is not a number.
        """
        if not isinstance(seconds_since_round_started, (float, int)):
            raise ValueError(f"seconds_since_round_started supports numbers, not {type(seconds_since_round_started)}!")

        if nodeid is None: nodeid = self.current_nodeid
        now_node = self.nodes[nodeid] 

        if now_node.node_status() == 'after': return False

        node_ending_time = now_node.time[-1]

        if seconds_since_round_started > node_ending_time: 
            return True

        return False


    def apply_abdicated_response(self, nodeid: int, abdicated_response: response_type = None) -> None:
        """
        Invoked when a player fails to respond within the allotted time to a node.

        Arguments:
            • nodeid (int): The ID of the node to apply the abdicated response to.
            • abdicated_response (response_type): The abdicated response provided by the frontend.

        Raises:
            Exception: If an abdicated response is to be applied to a chance node.

        Side-effects:
            Modifies the 'choice' and 'prediction' properties of the node, applying random responses where necessary.
        """
        if nodeid is None: 
            nodeid = self.current_nodeid
        this_node = self.nodes[nodeid]

        if abdicated_response is not None:
            abdicated_nodeid = abdicated_response['metadata']['nodeid']
            if abdicated_nodeid != nodeid:
                raise Exception(f"Node id mismatch: nodeid != abdicated_nodeid -> {nodeid} != {abdicated_nodeid}")

        if this_node.choicetypeis("chance"):
            raise Exception(f"Node {nodeid}: Abdicated response cannot be applied to chance nodes!")

        "Abdicated responses can be created by the frontend or backend."
        frontend_provided_abdicated_response = abdicated_response is not \
            None and abdicated_response["metadata"]["nodeid"] == nodeid

        if frontend_provided_abdicated_response:
            print(f"Abdicated response sent from frontend!")
            player_num = abdicated_response["metadata"]["player_num"] #TODO check this
            if this_node.chooser[player_num]:
                this_choice = this_node.choice
                this_choice[player_num] = abdicated_response["data"]
                this_node.choice = this_choice
            elif this_node.predictor[player_num]:
                this_prediction = this_node.prediction
                this_prediction[player_num] = abdicated_response["data"]
                this_node.prediction = this_prediction
            else: return None

        while this_node.node_status() != "after":
            "While the node requires more responses, generate random abdicated responses."
            reaction_time = round(this_node.time[1] + 1, 3)

            previous_choice = this_node.choice
            previous_prediction = this_node.prediction

            changed_choice, changed_prediction = False, False
            for player_num in range(self.nplayers):
                if this_node.chooser[player_num]:
                    if previous_choice[player_num] is None:
                        changed_choice = True
                        self.emit_abdication(player_number=player_num, response_type="choice")
                        self.players[player_num]['cumulative_payoffs'] -= self.abdication_penalty
                        previous_choice[player_num] = {'option': self._randomly_selected_option(parent_nodeid=nodeid, player_num=player_num), 
                                                      'keypress': None, 'rtimedn': reaction_time, 'rtimeup': reaction_time, 'timestamp': time.time()}
                elif this_node.predictor[player_num]:
                    if previous_prediction[player_num] is None:
                        changed_prediction = True
                        self.emit_abdication(player_number=player_num, response_type="prediction")
                        self.players[player_num]['cumulative_payoffs'] -= self.abdication_penalty
                        previous_prediction[player_num] = {'option': self._randomly_selected_option(parent_nodeid=nodeid, player_num=player_num),
                                                          'keypress': None, 'rtimedn': reaction_time, 'rtimeup': reaction_time, 'timestamp': time.time()}

            if changed_choice: this_node.choice = previous_choice
            if changed_prediction: this_node.prediction = previous_prediction


    def emit_abdication(self, player_number: int, response_type: str = None):
        """
        Emits to a player that they abdicated a response.  This triggers a feedback sound.

        Note: This only affects the sound, nothing else. 
        """
        if response_type not in ["choice", "prediction"]:
            response_type = None

        if self.experiment is not None:
            if self.experiment.experiment_type != 'tutorial':
                if self.experiment.experiment_manager is not None:
                    player = self.players[player_number]
                    if player["player_type"] != "robot":
                        penalty = self.abdication_penalty
                        player_sid = self.experiment.experiment_manager.uuids_to_sids[player["uuid"]]
                        message = f"You abdicated your {response_type if response_type is not None else 'response'}" 
                        message += f" and were deducted {penalty} {'payoff.' if penalty == 1 else 'payoffs.'}"
                        emit('abdication', {
                            'abdicated': True, 
                            'message': message,
                            'player_uuid': player.get('uuid', None), 
                            'penalty': penalty
                            }, to=player_sid)

                        if player_sid is None:
                            "If the player is not active on the website, convert them into a robot."
                            player["player_type"] = "robot"

                            for plr in self.experiment.players_list:
                                plr.player_type = 'robot'


    def response_to_tree(self, response: response_type = None, seconds_since_round_started: float = None, raise_exceptions: bool = False) -> bool:
        """
        Applies the response to the tree and returns if the tree is ready to be sent to the frontend.
        
        Arguments:
            • response: dict[str: dict[str: object]], optional; A nested dictionary that includes the response 
                itself and information about who submitted the response and where the response should be saved.
                example_response = {
                    'data': {
                        'option': 'BC', 
                        'keypress': 'd-space', 
                        'rtimedn': 5.076, 
                        'rtimeup': 6.147, 
                        'timestamp': float
                        },
                    'metadata': {
                        'nodeid': 2, 
                        'round_room_batch': (33, 4, 2, 'timestamp'), 
                        'player_uuid': '123e4567-e89b-12d3-a456-426614174000', 
                        'player_num': 0
                        }
                    } 
            • seconds_since_round_started: float, optional; The number of seconds since the round started.
                Because the experiment platform is single-threaded, the trees do not have endogenous countdown
                timers that allow them to independently apply abdicated responses once the time limit for the 
                current node expires.  Thus, an external loop continually updates the number of seconds since
                the round started so that the tree can apply any abdicated responses when needed.  This input
                is only necessary if response is None.  If response is not None, then this input should match
                the reaction time within the response.

        Raises:
            • ValueError: If both inputs are None 
            • ValueError: If response is improperly formatted
            • ValueError: If response is intended for a non-current node 
            • ValueError: If response is from a player not contained in the tree
            • ValueError: If information set attribute is improperly formatted
            • ValueError: If simultaneous choice response not sent to apex node
            • ValueError: If attempted to overwrite pre-existing response 

        Returns: 
            • bool; If True, the current node id should be incrimented and the tree sent to the fontend.
        """

        if response is None and seconds_since_round_started is None:
            """If response is None, then seconds_since_round_started must be a float (seconds) and if 
            seconds_since_round_started is None, then response must be a this value can be inferred from response """
            violation_message = "Both inputs to response_to_tree cannot be None!"
            if raise_exceptions:
                raise ValueError(violation_message)
            print(violation_message)
            return False

        if not self.valid_response(response=response):
            violation_message = f"response {response} is improperly formatted!"
            if raise_exceptions:
                raise ValueError(violation_message)
            print(violation_message)
            return False

        if seconds_since_round_started is None:
            seconds_since_round_started = response['data']['rtimeup']

        now_node: Node = self.nodes[self.current_nodeid]

        if now_node.isleaf():
            print("response_to_tree should not be called on leaf nodes!", now_node)
            return False

        response_was_abdicated = self.someone_abdicated(\
            seconds_since_round_started=seconds_since_round_started, nodeid=now_node.idnum)
        if isinstance(self.timestamps["abdicate_node_time"], (int, float)): 
            response_was_abdicated = time.time() > self.timestamps["abdicate_node_time"]  #testing TODO choose the best abdication checker
        if response_was_abdicated: print(f"Abdication Detected new way {int(time.time())} > {int(self.timestamps['abdicate_node_time'])}")

        if now_node.choicetypeis("chance"):
            self.choice_by_chance_node()
            return True

        if response_was_abdicated:
            self.apply_abdicated_response(\
                nodeid=self.current_nodeid, abdicated_response=response)
            return True        
        
        if response is not None:
            "Apply on-time responses to choice nodes."
            
            if response["metadata"]["round_room_batch"][0] != self.round_room_batch[0] or \
                response["metadata"]["round_room_batch"][1] != self.round_room_batch[1]:
                violation_message = f"Response sent to wrong tree. {response} - {self}."
                if raise_exceptions:
                    raise Exception(violation_message)
                print(violation_message)
                return False

            resp_nodeid = response["metadata"]["nodeid"]
            if resp_nodeid != self.current_nodeid:
                violation_message = f"Current node {self.current_nodeid}. Attempted to"
                violation_message += f" apply response to node {resp_nodeid}!"
                if raise_exceptions:
                    raise ValueError(violation_message)
                print(violation_message)
                return False
            
            player_uuid = response["metadata"]["player_uuid"]
            uuids_in_tree = [plr["uuid"] for plr in self.players]
            if player_uuid not in uuids_in_tree or response["metadata"]["player_num"] > len(uuids_in_tree): 
                violation_message = f"Response containing uuid {player_uuid} was sent to tree"
                violation_message += f" {self.title} which does not contain this player!"
                if raise_exceptions:
                    raise ValueError(violation_message)
                print(violation_message)
                return False

            if now_node.choicetypeis("simultaneous"):
                if len(now_node.info_set) < 2 or len(now_node.info_set[0]) > 1:
                    violation_message = f"Information set improperly formatted at node {self.current_nodeid}!"
                    if raise_exceptions:
                        raise ValueError(violation_message)
                    print(violation_message)
                    return False
                
                apex_node = now_node.info_set[0][0]
                if resp_nodeid != apex_node:
                    violation_message = f"Responses on simultaneous choice nodes must be sent to the apex {apex_node}! response {response}"
                    if raise_exceptions:
                        raise ValueError(violation_message)
                    print(violation_message)
                    return False
            
            violation_message = f"Temporal Paradox Averted!  Attempted to apply response to a"

            player_num = response["metadata"]["player_num"]    

            if now_node.chooser[player_num]:
                if isinstance(now_node.choice, tuple):
                    violation_message = f"{violation_message} choice attribute, which has already been solidified."
                    if raise_exceptions:
                        raise ValueError(violation_message)
                    print(violation_message)
                    return False
                
                "If the response is a choice, apply to choice and activate choice setter."
                temp_choice = now_node.choice
                if temp_choice[player_num] is not None:
                    print(f"Blocked redundant choice: node {now_node.idnum} - player {player_num}.")
                    return False
                temp_choice[player_num] = response["data"]
                now_node.choice = temp_choice
            else: 
                if isinstance(now_node.prediction, tuple):
                    violation_message = f"{violation_message} prediction attribute, which has already been solidified."
                    if raise_exceptions:
                        raise ValueError(violation_message)
                    print(violation_message)
                    return False
                
                "If the response is a prediction, apply to prediction and activate prediction setter."
                temp_prediction = now_node.prediction
                if temp_prediction[player_num] is not None:
                    print(f"Blocked redundant prediction: node {now_node.idnum} - player {player_num}.")
                    return False
                temp_prediction[player_num] = response["data"]
                now_node.prediction = temp_prediction
           
        if now_node.node_status() != "before":
            return True
        
        return False


    def tree_tag(self) -> None:
        """
        Generates a string attribute that uniquely identifies the game tree's structure and chooser positions. This
        attribute, known as the tree tag, essentially serves as a title for the game tree, enabling its categorization 
        within a database. The tree tag is created by iterating over all nodes and encoding their parent and chooser
        information in a binary string format.
        """
        parents_and_choosers = "TREE"
        for nodeid in self.nlst:
            this_node = self.nodes[nodeid]
            parent = str(this_node.parent)
            chooser = str(int("".join(\
                [str(int(chooser)) for chooser in this_node.chooser]), 2))
            node_string = parent + "." + chooser + "-"
            parents_and_choosers += node_string

        self.treetag = parents_and_choosers[:-1]


    def visual_permutation(self) -> None:
        """
        Determines the visual permutation of tree nodes based on the x-coordinates of child nodes. In the user-interface,
        the positioning of the game trees can be randomized to control for the impact of the position. This method is 
        utilized to preserve the random positioning of child nodes in the game tree for later reference or reproduction.
        """
        permutation = {}
        for nodeid in self.nlst:
            this_node = self.nodes[nodeid]
            childids, xcoords = [], []
            if not this_node.isleaf():
                for child in this_node.options:
                    xcoords.append(child.positionxy[0])
                    childids.append(child.idnum)
                childids = [x for _, x in sorted(zip(\
                    xcoords, childids), key=lambda pair: pair[0])]
                permutation[nodeid] = childids

        self.permutation = permutation
    

    def earnings(self, abdication_penalty: int = None) -> list[int]:
        """
        The cumulative payoffs at the current node of the tree.

        Returns:
            • list[int]: A list containing the cumulative earnings of each player.
        """
        return self._root.nodes[self._root.current_nodeid].cumulative_payoffs(abdication_penalty=abdication_penalty)
    

    @classmethod
    def total_earnings(cls, list_of_trees: list['Tree'], list_of_uuids: list[str], cents_per_payoff: float) -> dict[str: float]:
        """
        Computes the total dollar amount won by each player across multiple game trees.

        Arguments:
            • list_of_trees: list['Tree']; A list of game tree instances.
            • list_of_uuids: list[str]; A list of user IDs associated with the players.
            • cents_per_payoff: float; The conversion rate from payoffs to cents.

        Returns:
            • dict[str: float]: A dictionary mapping each player's user ID to their total earnings in cents.
        """
        earnings_dict = {uuid: 0.0 for uuid in list_of_uuids}

        for tree in list_of_trees:
            uuids = [player["uuid"] for player in tree.players]
            earnings = tree.earnings()
            for uuid, cumulative_payoff in zip(uuids, earnings):
                earnings_dict[uuid] += cumulative_payoff
       
        for uuid in list_of_uuids:
            earnings_dict[uuid] *= cents_per_payoff

        return earnings_dict


    @staticmethod
    def column_names(trees_with_responses: list['Tree'], expanded_format: bool = False, organize_by_idnum: bool = False, 
                     include_analysis_columns: bool = False, abbreviate_column_names: bool = False) -> list[str]:
        """
        Produces a list of column names for the dataframe for this tree.

        Arguments:
            • trees_with_responses: list['Tree']; List of all trees at the end of the experiment.
            • expanded_format: bool (default False); Determines dataframe format.  Both formats are interchangable.
                - Expanded (True): There is one column for every node attribute.
                - Contracted (False): There is one column for every node.
            • organized_by_idnum: bool (default False); Determines if nodes are identified by their 'idnum' or 
                node 'label' attributes.  Label identifiers are recommended because they express the path from 
                the root to the node, which is standardized across trees with different structures.
                - Idnum (True): Nodes are identified by integer node id numbers.
                - Label (False): Nodes are identified by string node labels.
            • include_analysis_columns: If true, includes helper columns for standard data analysis.
            • abbreviate_column_names: If True (not recommended), the column names will be abbreviated.

        Returns:
            • columns: list[str]; List of column names used in the dataframe.
        """
        for arg in [expanded_format, organize_by_idnum, include_analysis_columns, abbreviate_column_names]:
            if not isinstance(arg, bool): raise TypeError(f"Use boolean instead of {type(arg)}.")

        "Extracting only unique trees"
        unique_titles: list[str] = []
        unique_trees: list[Tree] = []
        for tree in trees_with_responses:
            if tree.title not in unique_titles:
                unique_titles.append(tree.title)
                unique_trees.append(tree)

        "Finding the maximum number of players across unique trees"
        n_players_max = 1
        for tree in unique_trees:
            if tree.nplayers > n_players_max:
                n_players_max = tree.nplayers

        "Extracting set of all nodes across all unique trees"
        node_columns = set()
        for tree in unique_trees:
            node_identifiers = [node_identifier for node_identifier in tree.nodes.keys() 
                                if isinstance(node_identifier, int) == organize_by_idnum]
            for node_identifier in node_identifiers: node_columns.add(node_identifier)

        "Sorting node columns by node identifiers"
        node_columns = sorted(list(node_columns))

        "Begin the list of column names with the general column names, either with full or abbreviated column names"
        columns = list(general_columns.values()) if abbreviate_column_names else list(general_columns.keys())

        if expanded_format:
            "In expanded format, there columns for each player in each attribute of each node."
            attributes = list(attribute_columns.values()) if abbreviate_column_names else list(attribute_columns.keys())
            list_attrs = list(list_attr_columns.values()) if abbreviate_column_names else list(list_attr_columns.keys())

            "Adding attribute column names per node such as 'probability_BC', meaning the probability at node BC."
            for attribute in attributes:
                for node_col in node_columns:
                    columns += [f"{attribute}_{node_col}"]

            "Adding list attribute column names per node per player such as 'payoffs_BC_p1'--payoffs at node BC for player 1."
            for list_attr in list_attrs:
                for node_col in node_columns:
                    for player_num in range(n_players_max):
                        columns += [f"{list_attr}_{node_col}_p{player_num}"]

        else: columns += node_columns if abbreviate_column_names else [f"node_{node_col}" for node_col in node_columns]

        if include_analysis_columns:
            columns += list(analysis_columns.values()) if abbreviate_column_names else list(analysis_columns.keys()) 

        return columns


    def data_file_row(self, expanded_format: bool = False, organize_by_idnum: bool = False, 
                     include_analysis_columns: bool = False, abbreviate_column_names: bool = False,
                     experiment_configuration_dict: dict[str: any] = None) -> dict[str: any]:
        """
        Converts all data from a game tree into a row, which trees_list_to_dataframe uses to generate a dataframe.
        
        Arguments:
            • trees_with_responses: list['Tree']; List of all trees at the end of the experiment.
            • expanded_format: bool (default False); Determines dataframe format.  Both formats are interchangable.
                - Expanded (True): There is one column for every node attribute.
                - Contracted (False): There is one column for every node.
            • organized_by_idnum: bool (default False); Determines if nodes are identified by their 'idnum' or 
                node 'label' attributes.  Label identifiers are recommended because they express the path from 
                the root to the node, which is standardized across trees with different structures.
                - Idnum (True): Nodes are identified by integer node id numbers.
                - Label (False): Nodes are identified by string node labels.
            • include_analysis_columns: If true, includes helper columns for standard data analysis.
            • abbreviate_column_names: If True (not recommended), the column names will be abbreviated.

        Returns:
            • dict[str: any]; Maps column names to single values for a row of the dataframe.
        """
        "Making sure that the adjacency matrix is JSON serializable."
        self.adjacency_matrix = list(self.adjacency_matrix)

        "Extracting a data for columns that applies to the entire tree, rather than specific nodes."
        row_data_general = {column_name: self[column_name] if column_name in self else None for column_name in general_columns}
        try:
            row_data_general['round'], row_data_general['room'], row_data_general['batch'], row_data_general['timestamp'] = self.round_room_batch
        except ValueError:
            try:
                row_data_general['round'], row_data_general['room'], row_data_general['batch'] = self.round_room_batch
                row_data_general['timestamp'] = self.round_room_batch[-1]
            except ValueError:
                row_data_general['round'] = self.round_room_batch[0]
                row_data_general['room'] = self.round_room_batch[1]
                row_data_general['batch'] = self.round_room_batch[-1]
                row_data_general['timestamp'] = self.round_room_batch[-1] #HACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        row_data_general['idnum_label_map'] = str({idnum: self.nodes[idnum].label for idnum in self.nlst})
        row_data_general['player_types'] = [player['player_type'] for player in self.players]
        row_data_general['avatar_shapes'] = [player['avatar']['shape'] for player in self.players]
        row_data_general['avatar_colors'] = [player['avatar']['color'] for player in self.players]
        row_data_general['player_uuids'] = [player['uuid'] for player in self.players]
        row_data_general['durations'] = str({'max': self.duration('max'), 'min': self.duration('max'), 
                                             'avg': self.duration('avg'), 'root': int(self.time[1] - self.time[0])})
        row_data_general['cumulative_payoffs_experiment'] = [player['cumulative_payoffs'] for player in self.players]
        abdication_penalty = experiment_configuration_dict.get('payoff_dimensions', {}).get(
            'abdication_penalty', None) if experiment_configuration_dict else None
        row_data_general['cumulative_payoffs_tree'] = self.cumulative_payoffs(abdication_penalty=abdication_penalty)
        row_data_general['players_abdicated'] = self.players_who_abdicated()
        row_data_general['final_node'] = self.current_nodeid
        row_data_general['tree_status'] = self.tree_status()
        row_data_general['tree_tag'] = self.treetag

        node_identifiers = self.nlst if organize_by_idnum else self.olst
        
        if expanded_format:
            "In expanded format, there columns for each player in each attribute of each node."
            attributes = list(attribute_columns.keys())
            list_attrs = list(list_attr_columns.keys())
            row_data_attributes = {}
            
            "Adding attribute data per node."
            for attribute in attributes:
                for node_identifier in node_identifiers:
                    if attribute in self.nodes[node_identifier]:
                        key = f"{attribute_columns[attribute] if abbreviate_column_names else attribute}_{node_identifier}"
                        if attribute == 'options':
                            row_data_attributes[key] = [child.idnum if organize_by_idnum else \
                                                        child.label for child in self.nodes[node_identifier].options]
                        else: row_data_attributes[key] = self.nodes[node_identifier][attribute]
                    elif attribute == "choice_type":
                        row_data_attributes[f"{attribute}_{node_identifier}"] = self.nodes[node_identifier].choicetypeis()

            "Adding list attribute data per node per player."
            for list_attr in list_attrs:
                for node_identifier in node_identifiers:
                    for player_num in range(self.nplayers):
                        key = f"{list_attr_columns[list_attr]}_{node_identifier}_p{player_num}" if \
                            abbreviate_column_names else f"{list_attr}_{node_identifier}_p{player_num}"
                        plr_choice = self.nodes[node_identifier].choice[player_num]
                        plr_prediction = self.nodes[node_identifier].prediction[player_num]
                        if list_attr == 'payoffs':
                            row_data_attributes[key] = self.nodes[node_identifier][list_attr][player_num]
                        elif list_attr == 'choice':
                            row_data_attributes[key] = None if plr_choice is None else plr_choice['option'] 
                        elif list_attr == 'prediction':
                            row_data_attributes[key] = None if plr_prediction is None else plr_prediction['option'] 
                        elif list_attr == 'choice_data':
                            row_data_attributes[key] = None if plr_choice is None else {
                                'rtype': plr_choice['keypress'], 'rtimedn': plr_choice[
                                    'rtimedn'], 'rtimeup': plr_choice['rtimeup']}
                        elif list_attr == 'prediction_data':
                            row_data_attributes[key] = None if plr_prediction is None else {
                                'rtype': plr_prediction['keypress'], 'rtimedn': plr_prediction[
                                    'rtimedn'], 'rtimeup': plr_prediction['rtimeup']}
            row_data = row_data_general | row_data_attributes

            "Updating dictionary mapping full column names to abbreviated column names."
            abbreviation_dict = {**general_columns}

        else:
            "If contracted format, create column data for each node in its entirety."
            row_data_nodes = {f"node_{node_identifier}": self.nodes[node_identifier] for node_identifier in node_identifiers}
            for node_identifier in node_identifiers:
                options = [child.idnum if organize_by_idnum else child.label for child in self.nodes[node_identifier].options]
                row_data_nodes[f"node_{node_identifier}"]["options"] = options
                "Pandas dataframes need the node cells to be normal dictionaries."
                row_data_nodes[f"node_{node_identifier}"] = dict(row_data_nodes[f"node_{node_identifier}"])
            row_data = row_data_general | row_data_nodes
            
            "Updating dictionary mapping full column names to abbreviated column names."
            abbreviation_dict = {**general_columns, **{f"node_{node_identifier}": 
                                 node_identifier for node_identifier in node_identifiers}}


        if include_analysis_columns:
            "Extracting data for analysis columns"
            row_data_analysis = {column_name: None for column_name in analysis_columns}
            row_data_analysis['truthfulness'] = self.truthfulness(objective_payoffs=True, as_ratio=False, player_num='average')
            row_data_analysis['agency'] =       self.agency(      objective_payoffs=True, as_ratio=False, player_num='average')
            row_data_analysis['alignment'] =    self.alignment(   objective_payoffs=True)
            row_data.update(row_data_analysis)
            abbreviation_dict.update(analysis_columns)

        if abbreviate_column_names:
            "Abbreviating all keys in row data"
            for full_key, abbreviated_key in abbreviation_dict.items():
                if full_key != abbreviated_key:
                    row_data[abbreviated_key] = row_data.pop(full_key)

        return row_data


    @staticmethod
    def trees_list_to_dataframe(trees_with_responses: list['Tree'], expanded_format: bool = False, organize_by_idnum: bool = False, 
                                include_analysis_columns: bool = False, abbreviate_column_names: bool = False, 
                                experiment_configuration_dict: dict[str: any] = None) -> pd.DataFrame:
        """
        At the end of an experiment or simulation, this function converts a list of game trees into a Pandas dataframe.

        Arguments:
            • trees_with_responses: list['Tree']; List of all trees at the end of the experiment.
            • expanded_format: bool (default False); Determines dataframe format.  Both formats are interchangable.
                - Expanded (True): There is one column for every node attribute.
                - Contracted (False): There is one column for every node.
            • organized_by_idnum: bool (default False); Determines if nodes are identified by their 'idnum' or 
                node 'label' attributes.  Label identifiers are recommended because they express the path from 
                the root to the node, which is standardized across trees with different structures.
                - Idnum (True): Nodes are identified by integer node id numbers.
                - Label (False): Nodes are identified by string node labels.
            • include_analysis_columns: If true, includes helper columns for standard data analysis.
            • abbreviate_column_names: If True (not recommended), the column names will be abbreviated.
            • experiment_configuration_dict: dict[str: any]; Stores researcher-inputted experiment settings.

        Returns:
            • results_df: pd.DataFrame; Dataframe stores all experiment data for convenient data analysis.
        """
        list_of_trees = []
        for tree in trees_with_responses:
            if isinstance(tree, Tree):  list_of_trees.append(tree)
            elif isinstance(tree, dict): tree = Tree.from_dict(tree_dict=tree)
            else: raise ValueError(
                    f"trees_with_responses must contain Trees not {type(tree)}.")
        trees_with_responses = list_of_trees

        "Extracting the list of column names from all of the game trees."
        columns = Tree.column_names(trees_with_responses=trees_with_responses, 
                                    expanded_format=expanded_format, organize_by_idnum=organize_by_idnum, 
                                    include_analysis_columns=include_analysis_columns, abbreviate_column_names=abbreviate_column_names)
        
        if experiment_configuration_dict is None: 
            experiment_configuration_dict = {}
        "The number of rows must be long enough to fit the experiment settings."
        number_of_rows = max(len(trees_with_responses), len(list(experiment_configuration_dict.keys())))
        column_dict = {column: [None] * number_of_rows for column in columns}

        "Every row of the data frame stores the data from one game room/tree in one round."
        for idx, tree in enumerate(trees_with_responses):
            row_data = tree.data_file_row(expanded_format=expanded_format, organize_by_idnum=organize_by_idnum, 
                                          include_analysis_columns=include_analysis_columns, abbreviate_column_names=abbreviate_column_names,
                                          experiment_configuration_dict=experiment_configuration_dict)
            for column_name, row_datum in row_data.items():
                column_dict[column_name][idx] = row_datum

        "Storing the settings that the researcher used to configure the experiment in the first two columns."
        setting_keys_col = 'setting_keys' if abbreviate_column_names else 'experiment_setting_keys'
        setting_vals_col = 'setting_vals' if abbreviate_column_names else 'experiment_setting_values'
        for jdx, (setting_key, setting_val) in enumerate(experiment_configuration_dict.items()):
            if jdx <= number_of_rows:

                column_dict[setting_keys_col][jdx] = setting_key
                column_dict[setting_vals_col][jdx] = setting_val

        "Converting list dictionary into Pandas dataframe."
        results_df = pd.DataFrame(column_dict)
        results_df.set_index(['round', 'room', 'batch'])

        for column in results_df.columns:
            "Elimenating erroneous NaN cells."
            results_df[column].replace(to_replace=np.nan, value=None, inplace=True)

        return results_df


    @classmethod
    def get_data_file(self, file_path: str, file_name: str = "My_Multiplayer_Experiment", 
            file_type: str = ".csv", most_recent_batch_only: bool = True, print_: bool = False) -> pd.DataFrame | list[dict]:
        """
        Retrieves an experiment file stored in the specified format from the given path. The file can either 
        be a CSV or a JSON file. The file is expected to contain game tree data from a multiplayer experiment.

        Arguments:
            • file_path: str; The directory where the file is located.
            • file_name: str; The name of the file to be retrieved.
            • file_type: str; The type of the file (".csv" or ".json") (default is ".csv").
            • print_: bool; A flag indicating whether to print the retrieved data (default is False).

        Returns:
            • pd.DataFrame or list[dict]: The experiment data as a pandas dataframe if a CSV file was read,
                or a list of dictionaries if a JSON file was read.

        Raises:
            • ValueError: If the file_type is not supported.
            • Exception: If the file could not be found.
        """
        file_types = ['.csv', '.json']
        if file_type[0] != ".": file_type = "." + file_type

        for char in file_types: file_name.replace(char, "")
        file_name = file_name.replace(" ", "_") + file_type

        full_path = os.path.join(file_path, file_name)
        cannot_find = f"Cannot find {file_name} in {file_path}."

        if file_type == ".csv":
            if os.path.exists(full_path):
                experiment_df = pd.read_csv(full_path)
                if 'Unnamed: 0' in experiment_df.columns: 
                    del experiment_df['Unnamed: 0']

                if most_recent_batch_only:
                    "Finding the timestamp for the most recent batch"
                    most_recent_batch_number = max(list(experiment_df['batch']))
                    experiment_df = experiment_df.loc[experiment_df['batch'] == most_recent_batch_number]

                if print_: 
                    pd.set_option('display.max_colwidth', 10)
                    pd.set_option('display.max_columns', 24)
                    print(experiment_df)
                return experiment_df    
            else: raise Exception(cannot_find)
            
        elif file_type == ".json":
            if os.path.exists(full_path):
                list_of_trees = json.load(open(full_path))

                if most_recent_batch_only:
                    most_recent_batch_number = max([tree.get('round_room_batch', (0, 0, 0, 0))[2] for tree in list_of_trees])
                    list_of_trees = [tree for tree in list_of_trees if tree.get('round_room_batch', (0, 0, 0, 0))[2] == most_recent_batch_number]

                if print_: pp.pprint(list_of_trees)
                return list_of_trees
            else: raise Exception(cannot_find)

        else: raise ValueError(f"file_type must be in {file_types}, not {file_type}.")


    @classmethod
    def save_data_file(cls, experiment_data: list['Tree'] | pd.DataFrame, file_path: str, file_name: str = "My_Multiplayer_Experiment", merge_data: bool = False) -> None:
        """
        Saves the experiment data to a file. The file type is inferred based on the data structure provided. 
        If experiment_data is a list of Trees, the data is saved as a JSON file. If it is a pandas dataframe,  
        the data is saved as a CSV file.

        Arguments:
            • experiment_data: list[Tree] or pd.DataFrame; The experiment data to be saved. 
                This should either be a list of game tree instances or a pandas dataframe.
            • file_path: str; The directory where the file should be saved.
            • file_name: str; The name of the file to be saved.
            • merge_data: bool; Whether to merge with existing data.

        Raises:
            • ValueError: If the type of experiment_data is not supported.
        """
        file_name = file_name.replace(" ", "_")
        full_path = os.path.join(file_path, file_name)  

        if isinstance(experiment_data, list) and all(isinstance(tree, Tree) for tree in experiment_data):
            for tree in experiment_data:
                tree.adjacency_matrix = list(tree.adjacency_matrix)
            extension = ".json"
            full_path += extension
            if merge_data and os.path.exists(full_path):
                with open(full_path, "r") as file:
                    existing_data = json.load(file)
                if isinstance(existing_data, list):
                    experiment_data = existing_data + experiment_data
            dump_function = lambda file: json.dump(experiment_data, file, indent=4)

        elif isinstance(experiment_data, pd.DataFrame):
            extension = ".csv"
            full_path += extension
            if merge_data and os.path.exists(full_path):
                existing_data = pd.read_csv(full_path)
                experiment_data = pd.concat([existing_data, experiment_data], ignore_index=True)
            dump_function = lambda file: experiment_data.to_csv(file, line_terminator='\n', index=False)

        else:
            raise ValueError(f"{type(experiment_data)} not supported! experiment_data must be a list of trees or a pandas dataframe.")

        base_file_name = file_name if not file_name.endswith(extension) else file_name[: -len(extension)]
        counter = 1
        while True:
            file_name = f"{base_file_name}({counter}){extension}" if counter > 1 else f"{base_file_name}{extension}"
            try:
                with open(os.path.join(file_path, file_name), "w") as file:
                    dump_function(file)
                break
            except PermissionError:
                counter += 1
                print(f"Permission denied. Trying again with filename: {file_name}")
                if counter > 50:
                    raise Exception(f"Permission denied. Failed to save: {file_name}")


    @staticmethod
    def list_of_trees(trees_lst: list['Tree'] | list[dict]) -> list['Tree']:
        """
        Converts all elements in trees_lst into Tree instances if they are not already.
        This function is typically useful when retrieving a json file of trees with 
        responses so that the tree dictionaries can be recast as Tree instances. 

        Arguments:
            • trees_lst: list['Tree'] | list[dict]; List of game trees.

        Returns:
            • trees: list['Tree']; List of Tree instances.
        """
        trees = []
        for tree in trees_lst:
            if isinstance(tree, Tree): 
                trees.append(tree)
            elif isinstance(tree, dict): 
                try: trees.append(Tree.from_dict(tree_dict=tree))
                except: raise ValueError(f"Dictionary could not be converted into a Tree instance: {tree}")
            elif isinstance(tree, str):
                try: trees.append(Tree.from_json(file_path=file_path_gametrees, file_name=tree))
                except: raise ValueError(f"Tree could not be found with title: {tree}")
            else: raise ValueError(f"trees_lst must contain only instances of Tree or dict, not {type(tree)}.")

        return trees


    @staticmethod
    def pack_trees(list_of_trees: list['Tree'], reverse_process: bool = False) -> list['Tree']:
        """
        Converts a list of trees into a list of rounds where each round is a list of trees in the same round.
        """
        if reverse_process:
            return [tree for round_ in list_of_trees for tree in round_]

        list_of_trees = Tree.list_of_trees(trees_lst=list_of_trees)
        list_of_trees = sorted(list_of_trees, key=lambda tree: (tree.round_room_batch[0], tree.round_room_batch[1]))

        packed_trees = []
        for tree in list_of_trees:
            if len(packed_trees) > tree.round_room_batch[0]:
                packed_trees.append([])
            packed_trees[tree.round_room_batch[0]].append(tree)

        return packed_trees


    @staticmethod
    def trees_from_experiment(experiment_code: str, most_recent_batch_only: bool = True) -> list['Tree']:
        """
        Returns the list of trees from a finished experiment based on the experiment code

        Arguments:
            • experiment_code: str; Unique ID of experiment.
            • most_recent_batch_only: bool (default True); If true, will only retrieve 
                the trees from the most recent batch.  This helps reduce computational demand.

        Returns:
            • trees: list['Tree']; List of Tree instances.
        """
        file_path = os.path.join(file_path_data, "Results")
        file_name = f"Experiment_Data_File_{experiment_code}"
        raw_data = Tree.get_data_file(file_path=file_path, file_name=file_name, 
                                      file_type=".json", most_recent_batch_only=most_recent_batch_only)
        return Tree.list_of_trees(trees_lst=raw_data)


    @staticmethod
    def trees_from_file() -> dict[str: int | float]:
        """
        Extracts all the tree titles within the folder that stores 
        them and maps them to a dictionary of their attributes.

        Arguments:
            • placeholder

        Returns:
            • trees: dict[str: int | float]; Flat dictionary mapping tree titles to a dict of their attributes
        """
        def tree_category(title: str):
            for category in ['BDG', 'ToM', 'Trust', 'Concealed', 'Gambling', 'Moral_Shielding', 'Tutorial']:
                if category in title:
                    return category
            return 'Named'

        try: 
            titles = [
                tree for tree in os.listdir(file_path_gametrees) 
                if os.path.isfile(os.path.join(file_path_gametrees, tree))
                ]
            trees = [
                Tree.from_json(file_path=file_path_gametrees, file_name=title) 
                for title in titles
                ]

            tree_attrs = {
                tree.title: {
                    'category': tree_category(tree.title), 
                    'depth': round(tree.depth_stats().get('mean'), 2),
                    'truthfulness': round(np.average(tree.truthfulness()), 2),
                    'agency': round(np.average(tree.agency()), 2),
                    'alignment': round(tree.alignment(), 2),
                    'nplayers': tree.nplayers, 
                    }
                for tree in trees
                }

            return {'trees': tree_attrs, 'success': True, 'message': 'Successfully extracted trees from folder.' }

        except Exception as e:
            error_message = f"Failed to extract trees from folder. Error: {type(e).__name__}: {e}"
            print(error_message)
            return {'trees': {}, 'success': False, 'message': error_message}
    

    @staticmethod
    def players_from_experiment(experiment_code: str) -> dict[str: play.ParticipantsData.Player]:
        """
        Returns the list of players from a finished experiment based on the experiment code

        Arguments:
            • experiment_code: str; Unique ID of experiment.

        Returns:
            • players: dict[str: dict]; List of Tree instances.
        """
        list_of_trees = Tree.trees_from_experiment(experiment_code=experiment_code)
        return {player['uuid']: player for players_at_root in 
                [tree.players for tree in list_of_trees] for player in players_at_root}


    @staticmethod
    def player_payoffs_by_round(experiment_code: str, cumulative: bool = False) -> dict[str: list[int]]:
        """
        Returns the cumulative payoffs per player per round for a finished experiment

        Arguments:
            • experiment_code: str; Unique ID of experiment.

        Returns:
            • player_payoffs: dict[str: list[int]]:; List of Tree instances.
        """
        list_of_trees = Tree.trees_from_experiment(experiment_code=experiment_code)
        players = Tree.players_from_experiment(experiment_code=experiment_code)
        n_rounds = list_of_trees[-1].round_room_batch[0] + 1
        player_payoffs = {'rounds': list(range(1, n_rounds+1))}

        for player_uuid in players.keys():
            player_payoffs[player_uuid] = [None] * n_rounds

        for tree in list_of_trees:
            round_num = tree.round_room_batch[0]
            earnings = tree.earnings()
            for idx, player in enumerate(tree.players):
                player_payoffs[player['uuid']][round_num] = earnings[idx]

        if cumulative:
            for player_uuid in players.keys():
                payoffs = player_payoffs[player_uuid]
                payoffs = [payoff if isinstance(payoff, (int, float)) else 0 for payoff in payoffs]
                player_payoffs[player_uuid] = [sum(payoffs[:idx + 1]) for idx in range(len(payoffs))]

        return player_payoffs


    @staticmethod
    def player_histories_abstracted(trees_with_responses: list['Tree'], exclude_abdications: bool = True) -> dict:
        """
        Abstracts player interaction histories from game trees, creating a concise representation of their choices.

        This function processes a list of game trees and extracts essential information about player interactions,
        including their choices, predictions, and the corresponding outcomes. It compiles a dictionary where each key
        is a tuple of player UUIDs representing a unique pair, and each value is a dictionary of lists that captures 
        the history of their interactions throughout the game.

        Arguments:
            trees_with_responses: A list of 'Tree' objects representing game trees with player responses.
            exclude_abdications: A boolean indicating whether to exclude interactions where a player abdicated.

        Returns:
            A dictionary mapping pairs of player UUIDs to their interaction histories. Each interaction history
            contains details such as round, level, time of interaction, choices, predictions, abdication status,
            and avatar information. This structured data is suitable for further analysis, such as calculating
            sequential choice probabilities.

        The output is structured as follows:
            {
                (player_uuid1, player_uuid2): {
                    'round_level_time': [(round_number, level, time), ...],
                    'response_p1': {
                        'label': [node_label, ...],
                        'pds': [pds_value, ...],
                        'pdo': [pdo_value, ...],
                        'sod': [sod_value, ...]
                    },
                    'response_p2': { ... },  # Similarly structured as 'response_p1'
                    'abdications': [(bool, bool), ...],
                    'chooser': [(bool, bool), ...],
                    'players': [{'shape': str, 'color': str, 'player_type': str}, ...]
                },
                ...
            }
        """
        histories = Tree.player_histories(trees_with_responses)
        experiment_start_time = trees_with_responses[0]['timestamps']['round_started_time']

        sequences = {}
        for pair, history in histories.items():
            avatar_info = None
            sequence = {
                'round_level_time': [],
                'response_p1': {'label': [], 'pds': [], 'pdo': [], 'sod': []},
                'response_p2': {'label': [], 'pds': [], 'pdo': [], 'sod': []},
                'abdications': [],
                'chooser': []
            }

            for tree in history:
                tree: Tree
                uuids = [player['uuid'] for player in tree.players]
                player_indices = (uuids.index(pair[0]), uuids.index(pair[-1]))
                round_started_time = tree['timestamps']['round_started_time']
                seconds_into_experiment = int(round_started_time - experiment_start_time)

                if avatar_info is None:
                    "Assuming avatar information does not change throughout the game"
                    player_1 = tree.players[player_indices[0]]
                    player_2 = tree.players[player_indices[1]]
                    avatar_info = [
                        {
                            'shape': player_1['avatar']['shape'],
                            'color': player_1['avatar']['color'],
                            'player_type': player_1['player_type']
                        },
                        {
                            'shape': player_2['avatar']['shape'],
                            'color': player_2['avatar']['color'],
                            'player_type': player_2['player_type']
                        }
                    ]

                for nodeid in tree.timeline():
                    node: Node = tree.nodes[nodeid]
                    abdications = node.players_who_abdicated()

                    if any(node.chooser) and node.node_status() == 'after' and (exclude_abdications or not any(abdications)):
                        sequence['chooser'].append((node.chooser[player_indices[0]], node.chooser[player_indices[-1]]))
                        sequence['abdications'].append((abdications[player_indices[0]], abdications[player_indices[-1]]))
                        sequence['round_level_time'].append((tree.round_room_batch[0], node.level, seconds_into_experiment))

                        response_p1 = node.choice[player_indices[0]] if node.chooser[player_indices[0]] else node.prediction[player_indices[0]]
                        response_p2 = node.choice[player_indices[1]] if node.chooser[player_indices[1]] else node.prediction[player_indices[1]]

                        selected_node_1 = tree.nodes[response_p1['option']]
                        selected_node_2 = tree.nodes[response_p2['option']]

                        payoff_dimensions_node_1 = selected_node_1.payoff_dimensions()
                        payoff_dimensions_node_2 = selected_node_2.payoff_dimensions()

                        for p_num, response, selected_node, payoff_dimensions in zip(
                                ['p1', 'p2'],
                                [response_p1, response_p2],
                                [selected_node_1, selected_node_2],
                                [payoff_dimensions_node_1, payoff_dimensions_node_2]):
                            sequence['response_' + p_num]['label'].append(response['option'])
                            sequence['response_' + p_num]['pds'].append(payoff_dimensions['intertemporal'][player_indices[0]])
                            sequence['response_' + p_num]['pdo'].append(payoff_dimensions['intertemporal'][player_indices[1]])
                            sequence['response_' + p_num]['sod'].append(payoff_dimensions['interpersonal'][player_indices[0]])

            sequences[pair] = {'players': avatar_info, **sequence}

        return sequences

    @staticmethod
    def create_bins(pds_set: set, pdo_set: set, sod_set: set) -> dict:
        "Creates optimally sized histogram bins from the set of unique data points"
        pds_lst = sorted(list(pds_set))
        pdo_lst = sorted(list(pdo_set))
        sod_lst = sorted(list(sod_set))

        axes_range = [
            min(pds_lst[0], pdo_lst[0], sod_lst[0]), max(pds_lst[-1], pdo_lst[-1], sod_lst[-1])
            ]            

        min_diff = axes_range[1] - axes_range[0]
        for dimension in [pds_lst, pdo_lst, sod_lst]:
            for val1, val2 in zip(dimension, dimension[1:]):
                diff = abs(val2 - val1)
                if diff < min_diff:
                    min_diff = diff

        start = axes_range[0] - min_diff / 2
        stop  = axes_range[1] + min_diff / 2
        steps = int((axes_range[1] - axes_range[0]) / min_diff + 2)
        limits = list(np.linspace(start, stop, steps))

        midpoints = [round((bucket[1] + bucket[0]) / 2, 4) for bucket in zip(limits, limits[1:])]

        bins = {midpoint: index for index, midpoint in enumerate(midpoints)}

        return {'pds': bins, 'pdo': bins, 'sod': bins, 'min_diff': min_diff}

    @staticmethod
    def find_bins(coordinates: tuple[float, float, float], bin_dicts: dict, return_type: str = 'midpoint') -> tuple[float | int, float | int, float | int]:

        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3 or any(not isinstance(val, float) for val in coordinates):
            raise ValueError(f'coordinates must be a tuple of tree floats, representing 3D payoff coordinates.')

        def find_bin(coordinate: float, dimension: str, min_diff=bin_dicts['min_diff']):

            for midpoint, index in bin_dicts[dimension].items():
                if midpoint - min_diff/2 <= coordinate < midpoint + min_diff/2:
                    return midpoint, index
            return None, None   

        getter = 1 if return_type == 'index' else 0       

        return tuple([find_bin(coord_dim[0], coord_dim[1])[getter] 
                        for coord_dim in zip(coordinates, ['pds', 'pdo', 'sod'])])  


    @staticmethod
    def sequential_choice_probabilities(trees_with_responses: list['Tree'], exclude_abdications = True, save_figure: bool = True) -> dict[str: list[float | str]]:
        """
        Generates and visualizes the probability of each future choice given the history of prior choices.
        
        This function iterates over a set of game trees and builds a probability model for predicting future choices
        based on the sequence of past choices and outcomes. It uses a trie structure to efficiently store and retrieve
        sequences of choices and their corresponding outcomes.

        Arguments:
            • trees_with_responses: A list of 'Tree' objects representing game trees with player responses.
            • exclude_abdications: A boolean indicating whether to exclude interactions where a player abdicated.
            • save_figure: A boolean indicating whether to save a visualization of the probabilities.

        Returns:
            • A dictionary mapping player UUIDs to their probability models, which predict future choices based
                on past interactions. The model is structured as a trie where each node represents a choice, and
                each leaf node stores the probabilities of possible outcomes following the sequence of choices 
                leading to that node.
        """
        class TrieNode:
            "Holds the sequences and outcomes"
            def __init__(self):
                self.children = {}
                self.outcomes = []

        def insert_sequence(node, sequence, outcome):
            for choice in sequence:
                if choice not in node.children:
                    node.children[choice] = TrieNode()
                node = node.children[choice]
            node.outcomes.append(outcome)

        def calculate_probabilities(node, sequence=()):
            "This is a helper function to convert the recursive generator into a dictionary."
            probabilities = {}
            if node.outcomes:
                outcome_counts = Counter(node.outcomes)
                total_outcomes = sum(outcome_counts.values())
                for outcome, count in outcome_counts.items():
                    probabilities[outcome] = count / total_outcomes
                return {sequence: probabilities}
            else:
                for choice, child_node in node.children.items():
                    new_sequence = sequence + (choice,)
                    probabilities.update(calculate_probabilities(child_node, new_sequence))
            return probabilities

        sequences = Tree.player_histories_abstracted(trees_with_responses=trees_with_responses, exclude_abdications=True)

        "Creating optimally sized histogram bins"
        pds_set, pdo_set, sod_set = set(), set(), set()
        for sequence in sequences.values():
            response_p1, response_p2 = sequence['response_p1'], sequence['response_p2']
            for pds in response_p1['pds'] + response_p2['pds']: pds_set.add(pds)
            for pdo in response_p1['pdo'] + response_p2['pdo']: pdo_set.add(pdo)
            for sod in response_p1['sod'] + response_p2['sod']: sod_set.add(sod)

        bins = Tree.create_bins(pds_set, pdo_set, sod_set)
        bin_permutations = list(it.permutations(bins['pds'].keys(), 3))
        possibilities = [(*perm, 'self') for perm in bin_permutations] + [(*perm, 'other') for perm in bin_permutations]

        player_uuids = sorted(list(set([pair[0] for pair in sequences.keys()])))
        prior_responses = {(possibility,): [] for possibility in possibilities}
        all_player_sequences = {uuid: prior_responses for uuid in player_uuids}

        for pair, sequence in sequences.items():
            player_uuid = pair[0]
            n_encounters = len(sequence['chooser'])
            added_self, added_other = False, False
            for idx in range(n_encounters):
                if idx < n_encounters - 1:
                    pds_, pdo_, sod_ = sequence['response_p1']['pds'][idx], \
                        sequence['response_p1']['pdo'][idx], sequence['response_p1']['sod'][idx]

                    pds, pdo, sod = Tree.find_bins((pds_, pdo_, sod_), bins, 'midpoint')

                    if sequence['chooser'][idx][0]:
                        added_self = True
                        key = ((pds, pdo, sod, 'self'),)
                        if player_uuid in all_player_sequences and key in all_player_sequences[player_uuid]:
                            all_player_sequences[player_uuid][key].append((
                                sequence['response_p1']['pds'][idx+1], 
                                sequence['response_p1']['pdo'][idx+1], 
                                sequence['response_p1']['sod'][idx+1],
                                'self' if sequence['chooser'][idx+1][0] else 'other'                        
                            ))

                    if sequence['chooser'][idx][1]:
                        added_other = True
                        key = ((pds, pdo, sod, 'other'),)
                        if player_uuid in all_player_sequences and key in all_player_sequences[player_uuid]:
                            all_player_sequences[player_uuid][key].append((
                                sequence['response_p1']['pds'][idx+1], 
                                sequence['response_p1']['pdo'][idx+1], 
                                sequence['response_p1']['sod'][idx+1],
                                'self' if sequence['chooser'][idx+1][1] else 'other'                              
                            ))

                    if added_self and added_other:
                        break

        history_outcome_pairs = {uuid: [] for uuid in player_uuids}
        for uuid in player_uuids:
            psequences = all_player_sequences[uuid]
            for history, outcomes in psequences.items():
                if outcomes:
                    history_outcome_pairs[uuid].append(tuple([(history, outcome) for outcome in outcomes]))

        root = TrieNode()
        probabilities = {}
        for uuid in player_uuids:
            for interaction in history_outcome_pairs[uuid]:
                for sequence, outcome in interaction:
                    insert_sequence(root, sequence, outcome)

            probability_model = calculate_probabilities(root)
            probabilities[uuid] = probability_model
            print(uuid)
            pp.pprint(probabilities[uuid])
            print(""), print("")

        return probabilities



    @staticmethod
    def sequential_choice_probabilities(trees_with_responses: list['Tree'], exclude_abdications = True, save_figure: bool = True) -> dict[str: list[float | str]]:
        """
        Generates and visualizes the probability of each future choice given the history of prior choices.
        
        This function iterates over a set of game trees and builds a probability model for predicting future choices
        based on the sequence of past choices and outcomes. It uses a trie structure to efficiently store and retrieve
        sequences of choices and their corresponding outcomes.

        Arguments:
            • trees_with_responses: A list of 'Tree' objects representing game trees with player responses.
            • exclude_abdications: A boolean indicating whether to exclude interactions where a player abdicated.
            • save_figure: A boolean indicating whether to save a visualization of the probabilities.

        Returns:
            • A dictionary mapping player UUIDs to their probability models, which predict future choices based
                on past interactions. The model is structured as a trie where each node represents a choice, and
                each leaf node stores the probabilities of possible outcomes following the sequence of choices 
                leading to that node.
        """
        class TrieNode:
            "Holds the sequences and outcomes"
            def __init__(self):
                self.children = {}
                self.outcomes = []

        def insert_sequence(node, sequence, outcome):
            for choice in sequence:
                if choice not in node.children:
                    node.children[choice] = TrieNode()
                node = node.children[choice]
            node.outcomes.append(outcome)

        def calculate_probabilities(node, sequence=()):
            "This is a helper function to convert the recursive generator into a dictionary."
            probabilities = {}
            if node.outcomes:
                outcome_counts = Counter(node.outcomes)
                total_outcomes = sum(outcome_counts.values())
                for outcome, count in outcome_counts.items():
                    probabilities[outcome] = count / total_outcomes
                return {sequence: probabilities}
            else:
                for choice, child_node in node.children.items():
                    new_sequence = sequence + (choice,)
                    probabilities.update(calculate_probabilities(child_node, new_sequence))
            return probabilities

        sequences = Tree.player_histories_abstracted(trees_with_responses=trees_with_responses, exclude_abdications=True)

        "Creating optimally sized histogram bins"
        pds_set, pdo_set, sod_set = set(), set(), set()
        for sequence in sequences.values():
            response_p1, response_p2 = sequence['response_p1'], sequence['response_p2']
            for pds in response_p1['pds'] + response_p2['pds']: pds_set.add(pds)
            for pdo in response_p1['pdo'] + response_p2['pdo']: pdo_set.add(pdo)
            for sod in response_p1['sod'] + response_p2['sod']: sod_set.add(sod)

        bins = Tree.create_bins(pds_set, pdo_set, sod_set)
        bin_permutations = list(it.permutations(bins['pds'].keys(), 3))
        possibilities = [(*perm, 'self') for perm in bin_permutations] + [(*perm, 'other') for perm in bin_permutations]

        player_uuids = sorted(list(set([pair[0] for pair in sequences.keys()])))


        interaction_histories = {uuid: {} for uuid in player_uuids}
        for pair, sequence in sequences.items():
            player_uuid = pair[0]
            n_encounters = len(sequence['chooser'])

            for idx_last, idx_next in zip(range(n_encounters), range(n_encounters)[1:]):

                key = []
                for idx in range(idx_last):
                    bin = Tree.find_bins((
                        sequence['response_p1']['pds'][idx], 
                        sequence['response_p1']['pdo'][idx], 
                        sequence['response_p1']['sod'][idx]
                        ), bins, 'midpoint')
                    
                    if sequence['chooser'][idx][0]:
                        key.append(tuple(list(bin) + ['self']))

                    elif sequence['chooser'][idx][1]:
                        key.append(tuple(list(bin) + ['other']))

                key = tuple(key)

                if interaction_histories[player_uuid].get(key, None) is None:
                    interaction_histories[player_uuid][key] = []
                interaction_histories[player_uuid][key].append((
                    sequence['response_p1']['pds'][idx_next], 
                    sequence['response_p1']['pdo'][idx_next], 
                    sequence['response_p1']['sod'][idx_next],
                    'self' if sequence['chooser'][idx_next][0] else 'other'
                    ))

        history_outcome_pairs = {uuid: [] for uuid in player_uuids}
        for uuid in player_uuids:
            psequences = interaction_histories[uuid]
            for history, outcomes in psequences.items():
                if outcomes:
                    history_outcome_pairs[uuid].append(tuple([(history, outcome) for outcome in outcomes]))

        root = TrieNode()
        probabilities = {}
        for uuid in player_uuids:
            for interaction in history_outcome_pairs[uuid]:
                for sequence, outcome in interaction:
                    insert_sequence(root, sequence, outcome)

            probability_model = calculate_probabilities(root)
            probabilities[uuid] = probability_model
            # print(uuid)
            # pp.pprint(probabilities[uuid])
            # print(""), print("")

        return probabilities



                
    # @staticmethod
    # def sequential_choice_probabilities(trees_with_responses: list['Tree'], exclude_abdications: bool = True, save_figure: bool = True) -> dict:
    #     """
    #     Generates and visualizes the probability of each future choice given the history of prior choices.
        
    #     This function iterates over a set of game trees and builds a probability model for predicting future choices
    #     based on the sequence of past choices and outcomes. It uses a trie structure to efficiently store and retrieve
    #     sequences of choices and their corresponding outcomes.

    #     Arguments:
    #         • trees_with_responses: A list of 'Tree' objects representing game trees with player responses.
    #         • exclude_abdications: A boolean indicating whether to exclude interactions where a player abdicated.
    #         • save_figure: A boolean indicating whether to save a visualization of the probabilities.

    #     Returns:
    #         • A dictionary mapping player UUIDs to their probability models, which predict future choices based
    #             on past interactions. The model is structured as a trie where each node represents a choice, and
    #             each leaf node stores the probabilities of possible outcomes following the sequence of choices 
    #             leading to that node.
    #     """
    #     class TrieNode:
    #         "Holds the sequences and outcomes"
    #         def __init__(self):
    #             self.children = {}
    #             self.outcomes = []

    #     def insert_sequence(node, sequence, outcome):
    #         "Insert a sequence of choices and its outcome into the trie"
    #         for choice in sequence:
    #             if choice not in node.children:
    #                 node.children[choice] = TrieNode()
    #             node = node.children[choice]
    #         node.outcomes.append(outcome)

    #     def calculate_probabilities(node, sequence=()):
    #         "Calculates probabilities for each outcome given a sequence of choices"
    #         probabilities = {}
    #         if node.outcomes:
    #             outcome_counts = Counter(node.outcomes)
    #             total_outcomes = sum(outcome_counts.values())
    #             probabilities = {outcome: count / total_outcomes for outcome, count in outcome_counts.items()}
    #         else:
    #             for choice, child_node in node.children.items():
    #                 new_sequence = sequence + (choice,)
    #                 probabilities.update(calculate_probabilities(child_node, new_sequence))
    #         return probabilities

    #     def create_bins_from_sequences(sequences: dict) -> dict:
    #         "Creates optimally sized histogram bins from the set of unique data points"

    #         pds_set, pdo_set, sod_set = set(), set(), set()
    #         for sequence in sequences.values():
    #             response_p1, response_p2 = sequence['response_p1'], sequence['response_p2']
    #             for pds in response_p1['pds'] + response_p2['pds']: pds_set.add(pds)
    #             for pdo in response_p1['pdo'] + response_p2['pdo']: pdo_set.add(pdo)
    #             for sod in response_p1['sod'] + response_p2['sod']: sod_set.add(sod)

    #         pds_lst = sorted(list(pds_set))
    #         pdo_lst = sorted(list(pdo_set))
    #         sod_lst = sorted(list(sod_set))

    #         axes_range = [
    #             min(pds_lst[0], pdo_lst[0], sod_lst[0]), 
    #             max(pds_lst[-1], pdo_lst[-1], sod_lst[-1])
    #             ]            

    #         min_diff = axes_range[1] - axes_range[0]
    #         for dimension in [pds_lst, pdo_lst, sod_lst]:
    #             for val1, val2 in zip(dimension, dimension[1:]):
    #                 diff = abs(val2 - val1)
    #                 if diff < min_diff:
    #                     min_diff = diff

    #         start = axes_range[0] - min_diff / 2
    #         stop  = axes_range[1] + min_diff / 2
    #         steps = int((axes_range[1] - axes_range[0]) / min_diff + 2)
    #         limits = list(np.linspace(start, stop, steps))

    #         midpoints = [round((bucket[1] + bucket[0]) / 2, 4) for bucket in zip(limits, limits[1:])]

    #         bins = {midpoint: index for index, midpoint in enumerate(midpoints)}

    #         return {'pds': bins, 'pdo': bins, 'sod': bins, 'min_diff': min_diff}
   
    #     def find_bins(coordinates: tuple[float, float, float], bin_dicts: dict, return_type: str = 'midpoint') -> tuple[float | int, float | int, float | int]:
    #         """
    #         Finds the bins for a given set of coordinates.

    #         Arguments:
    #             coordinates: The continuous payoff coordinates.
    #             bin_dicts: The dictionary containing the binning information.

    #         Returns:
    #             A tuple of bin indices corresponding to the coordinates.
    #         """
    #         if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3 or any(not isinstance(val, float) for val in coordinates):
    #             raise ValueError(f'coordinates must be a tuple of tree floats, representing 3D payoff coordinates.')

    #         def find_bin(coordinate: float, dimension: str, min_diff=bin_dicts['min_diff']):

    #             for midpoint, index in bin_dicts[dimension].items():
    #                 if midpoint - min_diff/2 <= coordinate < midpoint + min_diff/2:
    #                     return midpoint, index
    #             return None, None   

    #         getter = 1 if return_type == 'index' else 0       

    #         return tuple([find_bin(coord_dim[0], coord_dim[1])[getter] 
    #                       for coord_dim in zip(coordinates, ['pds', 'pdo', 'sod'])])  


    #     def get_choice_bins(sequence: dict, idx: int, bins: dict) -> tuple:
    #         """
    #         Retrieves the discretized bin indices for a player's choice or prediction at a specific interaction index.

    #         Arguments:
    #             sequence: The interaction sequence containing the response data for a player.
    #             idx: The index of the interaction within the sequence.
    #             bins: The dictionary containing the binning information for discretizing the data.

    #         Returns:
    #             A tuple representing the discretized bin indices for the player's choice or prediction.
    #         """
    #         "Retrieve the continuous coordinates for the current choice or prediction"
    #         pds_, pdo_, sod_ = sequence['response_p1']['pds'][idx], \
    #                            sequence['response_p1']['pdo'][idx], \
    #                            sequence['response_p1']['sod'][idx]

    #         "Discretize the continuous coordinates using the provided bins"
    #         pds_bin, pdo_bin, sod_bin = find_bins((pds_, pdo_, sod_), bins)

    #         "Determine if the current choice is 'self' or 'other'"
    #         role = 'self' if sequence['chooser'][idx][0] else 'other'

    #         "Return the discretized bins along with the role"
    #         return (pds_bin, pdo_bin, sod_bin, role)

    #     def save_probability_figure(probability_model: dict):
    #         # Placeholder for the function to save the probability visualization
    #         pass

    #     # Extracting and organizing interaction histories using the player_histories_abstracted function
    #     sequences = Tree.player_histories_abstracted(trees_with_responses, exclude_abdications)

    #     # Prepare the binning system for discretizing continuous data
    #     bins = create_bins_from_sequences(sequences)

    #     sequences = Tree.player_histories_abstracted(trees_with_responses=trees_with_responses, exclude_abdications=True)
    #     bins = create_bins_from_sequences(sequences)
    #     probabilities = {}

    #     for pair, sequence in sequences.items():
    #         root = TrieNode()  # Instantiate a new TrieNode for each player
    #         player_uuid = pair[0]
            
    #         # Build the trie from the sequences for this specific player
    #         for idx in range(len(sequence['chooser']) - 1):
    #             current_choice = get_choice_bins(sequence, idx, bins)
    #             next_choice = get_choice_bins(sequence, idx + 1, bins)
    #             insert_sequence(root, current_choice, next_choice)
            
    #         # Calculate probabilities for the current player
    #         player_probabilities = calculate_probabilities(root)
    #         probabilities[player_uuid] = player_probabilities

    #         if save_figure:
    #             save_probability_figure(player_probabilities)  # Placeholder for actual save function

    #         # Print the probabilities for debugging
    #         print(player_uuid)
    #         pp.pprint(player_probabilities)
    #         print(""), print("")


    #     return probabilities










    @staticmethod
    def player_histories(trees_with_responses: list['Tree']) -> dict[tuple[str, str]: list['Tree']]:
        """
        Returns a dictionary mapping pairs of participants to lists of all the game trees they played together.

        Arguments:
            • trees_with_responses: list['Tree']; List of game trees.

        Returns:
            • histories: dict[tuple[str, str]: list['Tree']]; {
                (uuid1, uuid2): [tree1, tree2, tree3...],
                (uuid5, uuid1): [tree1, tree2, tree3...],
                (uuid6, uuid8): [tree1, tree2, tree3...],
            }
        """
        trees_with_responses = Tree.list_of_trees(trees_with_responses)
        
        "Alphabetically sorted list of all player uuids in the experiment."
        players_ = sorted(list(set([player['uuid'] for tree in trees_with_responses for player in tree.players])))

        "List of all pairwise player permutations."
        player_permutations = [(uuid1, uuid2) for uuid1 in players_ for uuid2 in players_]

        "List dictionary of all pairwise player interaction histories."
        player_histories = {pair: [] for pair in player_permutations if pair[0] < pair[-1]} 

        for tree in trees_with_responses:
            uuids = sorted([player['uuid'] for player in tree.players])
            uuid_combos = list(it.combinations(uuids, 2))   
            for uuid_combo in uuid_combos:
                history = player_histories.get(uuid_combo, None)
                if history is not None:
                    player_histories[uuid_combo].append(tree)                

        return player_histories


    @staticmethod
    def cooperation_phase_spaces(experiment_code: str, save_figure: bool = False, 
                                 identify_by_avatar: bool = True, minimum_interaction_length: int = 2) -> dict[str: list[float | str]]:
        """
        Generates and visualizes phase space diagrams for pairs of matched players within and between dilemmas/games.
        The phase space is a payoff space with dimensions: x-axis: δπ𝑖 - payoff difference for player 𝑖, δπ𝑗 - payoff
        difference for player 𝑗, Δπ𝑖 - payoff difference between players 𝑖 and 𝑗, where each axis is the sum of payoff
        differences across the entire interaction history.  The direction of the arrows charts the history of the inter-
        action as players choose to cooperate, defect, trust, break trust, etc.

        Arguments:
            • experiment_code: str; Unique identifier of experiment.
            • save_figure: bool; If True, saves figure.  Otherwise, just returns the data. 
            • identify_by_avatar: bool; If True, the figure will identify players by avatar shape, not uuids.
            • minimum_interaction_length: int; Excludes all player interactions less than this number.

        Returns:
            • cumulative_histories: dict[str: list[float | str]]; A dictionary containing all data used to produce the phase space.
        """
        "Extract list of game trees"
        trees_with_responses: list[Tree] = Tree.trees_from_experiment(experiment_code=experiment_code)

        "Alphabetically sorted list of all player uuids in the experiment."
        players_ = sorted(list(set([player['uuid'] for tree in trees_with_responses for player in tree.players])))

        "List of all pairwise player permutations."
        player_permutations = [(uuid1, uuid2) for uuid1 in players_ for uuid2 in players_]

        "List dictionary of all pairwise player interaction histories."
        player_histories = {pair: [] for pair in player_permutations if pair[0] < pair[-1]}

        for tree in trees_with_responses:
            uuids = sorted([player['uuid'] for player in tree.players])
            avatars = [player['avatar']['shape'] for player in tree.players]
            player_numbers = [uuids.index(player['uuid']) for player in tree.players]
            pnum_combos = list(it.combinations(player_numbers, 2)) 
            uuid_combos = list(it.combinations(uuids, 2))
            
            for nodeid in tree.timeline():
                node = tree.nodes[nodeid]
                if not node.isleaf() and nodeid > 0:
                    "Identifying the chooser at the parent node."
                    parent = tree.nodes[node.parent]            
                    chooser_num = -1 if parent.choicetypeis("chance") else parent._chooser_number()
                    chooser_avatar = "chance node" if chooser_num < 0 else avatars[chooser_num]

                    "Extracting the payoff dimensions."
                    payoff_dimensions = node.payoff_dimensions(belief_key_prefix=None)
                    intertemporal = payoff_dimensions['intertemporal']
                    interpersonal = payoff_dimensions['interpersonal']

                    "Storing the chooser, chooser color, and payoff dimensions."
                    for pnum_combo, uuid_combo in zip(pnum_combos, uuid_combos):
                        uuid_i, uuid_j = uuid_combo[0], uuid_combo[1]
                        player_i, player_j = pnum_combo[0], pnum_combo[1]
                        pdi, pdj, ijd = intertemporal[player_i], intertemporal[player_j], interpersonal[player_i]
                        color = 'hsl(0, 0%, 0%)' if chooser_num < 0 else tree.players[chooser_num]['avatar']['color']
                        avatar_i = tree.players[player_i]['avatar']['shape']
                        avatar_j = tree.players[player_j]['avatar']['shape']
                        player_histories[(uuid_i, uuid_j)].append({
                            'pdi': pdi, 'pdj': pdj, 'ijd': ijd, 
                            'chooser': chooser_avatar, 'color': color, 
                            'avatar_i': avatar_i, 'avatar_j': avatar_j
                            })                        

        "Iterating in order of shortest to longest interaction history."
        sorted_histories = [x for _, x in sorted(zip(
            [len(val) for val in player_histories.values()], list(player_histories.keys())
            ), key=lambda pair: pair[0])]

        cumulative_histories = {}
        for key in sorted_histories:
            val = player_histories[key]
            if len(val) > minimum_interaction_length:
                "Creating a list for each dimension to meet Plotly's desired format."
                cumulative_history = {'pdi': [], 'pdj': [], 'ijd': [], 'chooser': [], 
                                      'color': [], 'avatar_i': [], 'avatar_j': []}

                for idx in range(len(val)):
                    sum_pdi = sum([val[jdx]['pdi'] for jdx in range(idx)])
                    sum_pdj = sum([val[jdx]['pdj'] for jdx in range(idx)])
                    sum_ijd = sum([val[jdx]['ijd'] for jdx in range(idx)])

                    "Rescaling each payoff dimension to be cumulative."
                    cumulative_history['pdi'].append(round(val[idx]['pdi'] + sum_pdi, 3))
                    cumulative_history['pdj'].append(round(val[idx]['pdj'] + sum_pdj, 3))
                    cumulative_history['ijd'].append(round(val[idx]['ijd'] + sum_ijd, 3))
                    cumulative_history['avatar_i'].append(val[idx]['avatar_i'])
                    cumulative_history['avatar_j'].append(val[idx]['avatar_j'])
                    cumulative_history['chooser'].append(val[idx]['chooser'])
                    cumulative_history['color'].append(val[idx]['color'])

                "Normalizing values between -1 and 1."
                max_pdi = max([abs(val) for val in cumulative_history["pdi"]])
                max_pdj = max([abs(val) for val in cumulative_history["pdj"]])
                max_ijd = max([abs(val) for val in cumulative_history["ijd"]])
                max_dim = int(round(max(max_pdi, max_pdj, max_ijd) + 0.55, 3))
                cumulative_history["pdi_norm"] = [round(val / max_dim, 3) for val in cumulative_history["pdi"]]
                cumulative_history["pdj_norm"] = [round(val / max_dim, 3) for val in cumulative_history["pdj"]]
                cumulative_history["ijd_norm"] = [round(val / max_dim, 3) for val in cumulative_history["ijd"]]

                cumulative_histories[key] = cumulative_history
            else: del player_histories[key]
            
        if save_figure:
            fig, buttons = go.Figure(), []
            fig.update_layout(template=fig_layout["template"], font=fig_layout["font"])
            n_interactions = len(list(cumulative_histories.items()))
            xtitle = f'Payoff Difference for Player 𝑖'
            ytitle = f'Payoff Difference for Player 𝑗'
            ztitle = f'Players 𝑖 - 𝑗 Payoff Difference'
            line_color = "hsl(0, 0%, 100%)"

            def draw_phase_space(fig: go.Figure = fig):
                "Draws the phase space axes arrows"

                arrow_inputs = {'visible': True, 'showlegend': False, 'mode': 'markers', 'opacity': 1.0, 'marker': {'color': line_color, 'size': 10, 'symbol': 'diamond'}}
                fig.add_trace(go.Scatter3d(name=f'Arrowhead: {xtitle}', hovertemplate = f'Arrowhead: {xtitle}', **arrow_inputs, x=(1, 1), y=(0, 0), z=(0, 0)))
                fig.add_trace(go.Scatter3d(name=f'Arrowhead: {ytitle}', hovertemplate = f'Arrowhead: {ytitle}', **arrow_inputs, x=(0, 0), y=(1, 1), z=(0, 0)))
                fig.add_trace(go.Scatter3d(name=f'Arrowhead: {ztitle}', hovertemplate = f'Arrowhead: {ztitle}', **arrow_inputs, x=(0, 0), y=(0, 0), z=(1, 1)))

                aline_inputs = {'visible': True, 'showlegend': False, 'mode': 'lines', 'opacity': 0.9, 'line': {'color': line_color, 'width': 9}}
                fig.add_trace(go.Scatter3d(name=f'Axis: {xtitle}', hovertemplate = f'Axis: {xtitle}', **aline_inputs, x=(-1, 1), y=(0, 0), z=(0, 0)))
                fig.add_trace(go.Scatter3d(name=f'Axis: {ytitle}', hovertemplate = f'Axis: {ytitle}', **aline_inputs, x=(0, 0), y=(-1, 1), z=(0, 0)))
                fig.add_trace(go.Scatter3d(name=f'Axis: {ztitle}', hovertemplate = f'Axis: {ztitle}', **aline_inputs, x=(0, 0), y=(0, 0), z=(-1, 1)))

            draw_phase_space(fig=fig)

            "Storing a boolean list dictionary that tells Plotly which traces are visible when scrolling."
            meetings_per_history = [len(history['color']) for history in cumulative_histories.values()]
            n_histories, n_total_meetings = len(meetings_per_history), sum(meetings_per_history)
            visible_dict = {idx: [True] * 6 + [False] * n_total_meetings * 2 for idx in range(n_histories)}

            trace_idx = 6
            for idx, n_meetings in enumerate(meetings_per_history):
                visible_lst = visible_dict[idx]
                for meeting in range(n_meetings * 2):
                    visible_lst[trace_idx] = True
                    trace_idx += 1

            for idx, (pair, history) in enumerate(list(cumulative_histories.items())):
                uuid1_, uuid2_ = pair[0][:6] + "...", pair[1][:6] + "..."
                is_visible = True if idx == 0 else False
                
                for jdx in range(len(history['color'])):
                    pdi = [0 if jdx == 0 else history['pdi_norm'][jdx - 1], history['pdi_norm'][jdx]]
                    pdj = [0 if jdx == 0 else history['pdj_norm'][jdx - 1], history['pdj_norm'][jdx]]
                    ijd = [0 if jdx == 0 else history['ijd_norm'][jdx - 1], history['ijd_norm'][jdx]]
                    avatar_i, avatar_j = history['avatar_i'][jdx], history['avatar_j'][jdx]

                    "Determines if players are identified by uuids or avatar shapes."
                    identify_i = avatar_i if identify_by_avatar else uuid1_
                    identify_j = avatar_j if identify_by_avatar else uuid2_

                    line = dict(color=history['color'][jdx], width=9)
                    marker = dict(color=history['color'][jdx], size=11, symbol='circle')
                    hover_text = f"Meeting {jdx}:<br>𝑖: {identify_i}<br>𝑗: {identify_j}<br><br>Chooser:<br>{history['chooser'][jdx]}"
                    hover_text += f"<br>δπ𝑖: {history['pdi'][jdx]}<br>δπ𝑗: {history['pdj'][jdx]}<br>Δπ𝑖: {history['ijd'][jdx]}"
                    trace_inputs = {'name': '', 'visible': is_visible, 'showlegend': False, 'hovertemplate': hover_text}
                    fig.add_trace(go.Scatter3d(**trace_inputs, mode='markers', opacity=1.0,
                        marker=marker, x=(pdi[-1], pdi[-1]), y=(pdj[-1], pdj[-1]), z=(ijd[-1], ijd[-1])))
                    fig.add_trace(go.Scatter3d(**trace_inputs, mode='lines', opacity=0.9, line=line, x=pdi, y=pdj, z=ijd))

                buttons.append(dict(label=f"{idx}", method="update", args=[
                    {"visible": visible_dict[idx]}, {"title": f"Cooperation History: {identify_i} & {identify_j}"}
                    ]))

            "Slider allows user to scroll through each pairwise interaction history."
            sliders = [dict(active=0, font=dict(size=20), currentvalue={"prefix": "Game: "}, pad={"t": 50}, steps=buttons)]

            axis = {'visible': True, 'showticklabels': True, 'tickfont': {'size': 14}, 'nticks': 9, 'range': (-1.05, 1.05)}
            fig.update_scenes(xaxis_title=dict(text=xtitle, font=dict(size=20)), xaxis=axis)
            fig.update_scenes(yaxis_title=dict(text=ytitle, font=dict(size=20)), yaxis=axis)
            fig.update_scenes(zaxis_title=dict(text=ztitle, font=dict(size=20)), zaxis=axis)

            annotat_rem_inputs = {'z': 0, 'showarrow': False, 'font': {'color': line_color, 'size': 32}}
            fig.update_scenes(annotations=[
                dict(text="Win-win",   x= 0.5, y= 0.5, **annotat_rem_inputs),
                dict(text="Selfish",   x= 0.5, y=-0.5, **annotat_rem_inputs),
                dict(text="Malicious", x=-0.5, y=-0.5, **annotat_rem_inputs),
                dict(text="Helpful",   x=-0.5, y= 0.5, **annotat_rem_inputs)
                ]) 

            fig.update_layout(sliders=sliders, 
                title_text=f"Cooperation Phase Space", showlegend=False, 
                scene_camera=dict(eye=dict(x=-0.0, y=-0.00001, z=2)), scene_aspectmode='cube', 
                title_x=0.5, title_y=0.975, titlefont_size=50, hoverlabel=dict(font_size=22))
            
            # file_path = "C:/Users/Gregory Stanley/Desktop/U of M/Research Archive/Multiplayer/ABM_Simulation/Outputs/Exp_Visuals/" #TODO find perminent path
            file_path = "./server/game_engine/Data/Figures" 
            file_name = f"Cooperation_Phase_Space_{n_interactions}"
            experiment = trees_with_responses[0].experiment

            if experiment is not None:
                if isinstance(experiment, dict):
                    experimenter_uuid = experiment.get('experimenter_uuid', None)
                    experiment_code = experiment.get('experiment_code', None)
                    if experimenter_uuid and experiment_code:
                        file_name += f"{experimenter_uuid}~{experiment_code}"
                elif hasattr(experiment, 'experiment_configuration_dict') and hasattr(experiment, 'experiment_code'):
                    experimenter_uuid = experiment.experiment_configuration_dict['experimenter_uuid']
                    experiment_code = experiment.experiment_configuration_dict['experiment_code']
                    if experimenter_uuid and experiment_code:
                        file_name += f"{experimenter_uuid}~{experiment_code}"                   

            fig.write_html(os.path.join(file_path, file_name + '.html'))
        # return cumulative_histories
        return {str(key): val for key, val in cumulative_histories.items()}


    def to_dict(self, keys_to_include: list[str] = ["all"]) -> dict[str: list]:
        """
        Converts the game tree instance into a dictionary. The dictionary has attribute names as keys and lists of attribute
        values as values. Each list is indexed by node ID numbers. This format can be useful for inspecting the tree structure
        and node attributes in a structured, readable manner.

        Arguments:
            • keys_to_include: list[str]; A list of node attributes to include in the dictionary. Use ["all"] 
                to include all attributes, or ["?"] to print all available keys without transforming the tree.

        Returns:
            • dict[str: list]: A dictionary representation of the game tree.
                example_tree_list_dict = {
                    'idnum':   [ 0, 1, 2, 3, 4, 5, 6],
                    'parent':  [-1, 0, 0, 1, 1, 2, 2],
                    'label':   ['','A','B','AA','AB','BA','BB'],...
                    }
        """ 
        if keys_to_include[0] == "?":
            "Prints the keys so you can copy the ones you care about into keys_to_include."
            print(list(self.nodes[self.nlst[-1]].keys()))
            return None

        if keys_to_include[0] == "all": 
            keys = list(self.nodes[self.nlst[-1]].keys())
        else: keys = keys_to_include
        list_dict = {key: [] for key in keys}

        for nodeid in self.nlst:
            this_node = self.nodes[nodeid]
            for key in keys:
                if key == "options":
                    these_options = []
                    if not this_node.isleaf():
                        for child in this_node.options:
                            these_options.append((child.idnum, child.label))
                    list_dict[key].append(these_options)
                else: list_dict[key].append(this_node[key])

        list_dict['general'] = {'current_nodeid': self.current_nodeid, 'players': self.players, 'nplayers': self.nplayers, 
            'title': self.title, 'round_room_batch': self.round_room_batch, 'treetag': self.treetag, 'permutation': self.permutation}
        
        return list_dict
    

    def to_list(self, keys_to_include: list[str] = ["all"]) -> list[dict]:
        """
        Converts the game tree instance into a list of dictionaries, where each dictionary represents 
        a node in the tree.  This format can be useful for inspecting individual nodes in isolation or 
        for representing the tree in a format suitable for conversion to JSON.

        Arguments:
            • keys_to_include: list[str]; A list of node attributes to include in the dictionary. Use ["all"] 
                to include all attributes, or ["?"] to print all available keys without transforming the tree.

        Returns:
            • list[dict]: A list of dictionaries, each representing a node in the game tree.
                example_dict_list = [
                    {'idnum': 0, 'parent': -1,... 'options': [1, 2]},
                    {'idnum': 1, 'parent':  0,... 'options': []},
                    {'idnum': 2, 'parent':  0,... 'options': []}
                    ]
        """
        if keys_to_include[0] == "?":
            "Prints the keys so you can copy the ones you care about into keys_to_include."
            print(list(self.nodes[self.nlst[-1]].keys()))
            return None

        if keys_to_include[0] == "all": 
            keys = list(self.nodes[self.nlst[-1]].keys())
        else: keys = keys_to_include
        tree_lst = [{} for nodeid in self.nlst]

        for nodeid in self.nlst:
            new_node = tree_lst[nodeid]
            this_node = self.nodes[nodeid]
            for key in keys:
                if key == "options":
                    these_options = []
                    if not this_node.isleaf():
                        for child in this_node.options:
                            these_options.append((child.idnum, child.label))
                    new_node["options"] = these_options
                else: new_node[key] = this_node[key]

        return tree_lst


    def draw_probability(self) -> float:
        """
        Randomly samples the probability if the node's probability is ambiguous.

        Returns
            • float; The sampled probability.
        """
        if self.probability[1] == 1: return self.probability[0]
        if self.probability[1] == 0: return round(random.random(), 4)
        if self.probability[0] == 0.5: return round(random.uniform(\
            0.5 - (1 - self.probability[1]) / 2, 0.5 + (1 - self.probability[1]) / 2), 4)
        ambiguity_interval_lower = self.probability[0] - (1 - self.probability[1]) / 2
        ambiguity_interval_upper = self.probability[0] + (1 - self.probability[1]) / 2
        if ambiguity_interval_lower < 0: 
            ambiguity_interval_upper -= ambiguity_interval_lower
            ambiguity_interval_lower = 0
        elif ambiguity_interval_upper > 1: 
            ambiguity_interval_lower -= ambiguity_interval_upper - 1
            ambiguity_interval_upper = 1
        return round(random.uniform(\
            ambiguity_interval_lower, ambiguity_interval_upper), 4)


    def to_json(self, file_path: str) -> None:
        """
        Serializes the game tree instance to a JSON file and saves it to the specified file path.

        Arguments:
            • file_path: str; The directory where the JSON file should be saved.
        """
        file_name = self.title.replace(" ", "_")
        if file_name[-5:] != ".json": file_name += ".json"

        if isinstance(self.adjacency_matrix, match.AdjacencyMatrix):
            # self.adjacency_matrix = self.adjacency_matrix.json_serializable_matrix()
            self.adjacency_matrix = list(self.adjacency_matrix)

        if self.experiment is not None and not isinstance(self.experiment, dict):
            self.experiment = self['experiment'] = {
                'experiment_code': self.experiment.experiment_code, 
                'experimenter_uuid': self.experiment.experimenter_uuid
                }

        with open(os.path.join(file_path, file_name), "w") as file: 
            json.dump(self, file, indent=4)  
        print(f"Saved: {file_name}")


    @classmethod
    def from_dict(cls, tree_dict: dict) -> 'Tree':
        """
        Reconstructs a Tree object from a dictionary. 

        Arguments:
            • tree_dict: dict; The dictionary representation of a Tree.

        Returns:
            • Tree: A Tree instance that matches the structure and attributes of the dictionary input.
        """
        if isinstance(tree_dict, Tree): 
            return tree_dict

        def extract_parent_list(tree_dict: dict = tree_dict, parents: list[int] = None):
            """Recurses the dictionary to produce a list of parent id numbers."""

            if parents is None: parents = []

            for child in tree_dict["options"]:
                parents.append(child["parent"])
                extract_parent_list(tree_dict=child, parents=parents)

            return parents
        
        parents = sorted(extract_parent_list(tree_dict=tree_dict, parents=None))
        tree = cls(title=tree_dict["title"], nplayers=tree_dict["nplayers"], players=tree_dict["players"], 
                   avatar_colors=tree_dict["avatar_colors"], adjacency_matrix=tree_dict["adjacency_matrix"], 
                   timestamps=tree_dict["timestamps"], edges=parents) 
        tree.current_nodeid = tree_dict["current_nodeid"]
        experiment = tree_dict.get("experiment", None)
        if experiment is not None: tree.experiment = experiment

        def attr_setter(tree: Tree = tree, tree_dict: dict = tree_dict) -> Tree:
            """Sets all Tree and Node attributes based on the keys of tree_dict."""
            for key in tree_dict:
                if key != "options":
                    setattr(tree, f"{key}", tree_dict[f"{key}"])

            if tree.options and tree_dict["options"]:
                for tchild, dchild in zip(tree.options, tree_dict["options"]):
                    attr_setter(tree=tchild, tree_dict=dchild)
            "TODO Test importing Beliefs from json/dict"
            beliefs_ = copy.deepcopy(tree.beliefs)
            tree.beliefs = Beliefs(tree.idnum, tree._root)
            for belief_key in beliefs_:
                tree.beliefs[belief_key] = beliefs_[belief_key]

            return tree

        tree = attr_setter(tree = tree, tree_dict = tree_dict)

        return tree


    @classmethod
    def from_json(cls, file_path: str, file_name: str = "My_Game_Tree") -> 'Tree':
        """
        Reconstructs a Tree object from a JSON file.

        Arguments:
            • file_path: str; The directory where the JSON file is located.
            • file_name: str; The name of the JSON file and title of tree.

        Returns:
            • Tree: A Tree instance, with a title inferred from file_name

        Raises:
            • Exception: If the JSON file could not be found.
        """
        file_name = file_name.replace(" ", "_")
        if file_name[-5:] != ".json": file_name += ".json"
   
        full_path = os.path.join(file_path, file_name)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                tree_dict = json.load(f)
            return cls.from_dict(tree_dict)
        else: raise Exception(f"Cannot find {file_name} in {file_path}.")


    def dprint(self, keys_to_include: list[str] = ["all"]) -> None:
        "Pretty prints the tree by attribute using the to_dict method."
        if keys_to_include[0] == "?":
            "Prints the keys so you can copy the ones you care about into keys_to_include."
            print(list(self.nodes[self.nlst[-1]].keys()))
            return None
        pp.pprint(self.to_dict(keys_to_include=keys_to_include))


    def tprint(self, keys_to_include: list[str] = ['idnum', 'chooser', 'predictor', 'payoffs', 
            'parent', 'positionxy', 'probability', 'info_set', 'label', 'level', 'time', 'options']):
        """TODO UNFINISHED Prints tree such that it is easy to understand visually."""

        if keys_to_include[0] == "?":
            "Prints the keys so you can copy the ones you care about into keys_to_include."
            print(list(self.nodes[self.nlst[-1]].keys()))
            return None

        def color_string(fullstring: str, substring: str, color_integer: int = None, color_past_substring: tuple = None, 
                         just_make_bold: bool = False, just_underline: bool = False, print_: bool = False):
            """Colors a substring within a string to draw attention to the substring.
            
            Arguments
            • fullstring / substring: string; substring must be within fullstring.
            • color_integer: integer between 0 and 7; Determines the color of the substring
            • color_past_substring: tuple[int, int], like (4, 1); Will color the string
                at an index starting at end of substring + 4 and for a length of 1.
            • just_make_bold / just_underline: bool; Will bolden or underline, not color.
            • print_: bool; If True, will print the output instead of returning it."""

            fullstring, substring = str(fullstring), str(substring)
            if substring not in fullstring:
                fullstring, substring = f"substring '{substring}' MUST be in fullstring '{fullstring}'!", "MUST"

            idx = fullstring.find(substring)
            end, max_int = idx + len(substring), 7

            color_integer = color_integer % max_int if isinstance(color_integer, int) else random.randrange(1, max_int)

            if just_make_bold: new_str = f"{fullstring[:idx]}\033[1m{substring}\033[0m{fullstring[end:]}"
            elif just_underline: new_str = f"{fullstring[:idx]}\033[4m{substring}\033[0m{fullstring[end:]}"

            elif isinstance(color_past_substring, tuple) and len(color_past_substring) == 2:
                start, stop = color_past_substring
                new_str = f"{fullstring[:end + start]}\033[9{color_integer}m{fullstring[end+start:end+start+stop]}\033[0m{fullstring[end+start+stop:]}" 
            else: new_str = f"{fullstring[:idx]}\033[9{color_integer}m{substring}\033[0m{fullstring[end:]}" 
            
            if print_: print(new_str)
            return new_str

        "Printing general information about the tree."
        general_info = f"{self.title} - current node {self.current_nodeid} - "
        general_info = color_string(general_info, str(self.title), just_make_bold=True)
        player_str = str(self.players)
        for idx, player in enumerate(self.players):
            player_str = color_string(fullstring=player_str, substring=player, color_integer=idx+1)
        general_info += f"experiment round {self.round_room_batch[0]} - game room {self.round_room_batch[1]} - players {player_str}"
        print(general_info)

        max_nchoosers = 0
        for nodeid in self.nlst:
            if sum(self.nodes[nodeid].chooser) > max_nchoosers: 
                max_nchoosers = sum(self.nodes[nodeid].chooser)

        copy_tree = self.to_list()

        for nodeid, node in enumerate(copy_tree):
            node_str = f"{nodeid}:" if nodeid > 9 else f" {nodeid}:"
            nchoosers = sum(node["chooser"])
            padding = " " * (max_nchoosers - (nchoosers - 1) * 2 + 2)
            choo_str = padding[:-2] + "c" if nchoosers == 0 else padding
            for player_num, chooser in enumerate(node["chooser"]):
                if chooser: choo_str += color_string(fullstring=str(player_num), 
                            substring=str(player_num), color_integer=player_num+1)
                if chooser and nchoosers > 1 and player_num < nchoosers: choo_str += "&"
            node_str += choo_str
            print(node_str)


    def _updateids(self, parentid: int) -> int:
        """
        Determines the id number for the new node and updates the id numbers of existing nodes in the tree.

        This method is called during the process of adding a new node to the tree. It establishes the id number for the new
        node in a way that maintains the existing order of the nodes. The id number of the new node is determined based on 
        the ids of the parent node's peers and their children. Once the id for the new node is determined, the id numbers  
        of all younger nodes in the tree are updated to maintain the proper order.

        Arguments:
            • parentid: int; The id number of the parent node under which the new node is being added.

        Returns:
            • int: The id number to be assigned to the new node.
        """
        self.nlst = sorted([node for node in self.nodes if isinstance(node, int)])
        self.olst = [self.nodes[node].label for node in self.nlst]
        self.levels = [self.nodes[node].level for node in self.nlst]

        parent = self.nodes[parentid]
        peers_of_parent = self.relative_ages(nodeid=parentid, age_relation="peers of")
        peers_of_parent_with_kids = [0 if self.nodes[peer].isleaf() else 1 for peer in peers_of_parent]
        next_oldest_peer_with_kids = [par for par, has_kids in zip(peers_of_parent, 
            peers_of_parent_with_kids) if has_kids == 1 and par <= parent.idnum]
        next_oldest_peer_with_kids = None if len(\
            next_oldest_peer_with_kids) == 0 else next_oldest_peer_with_kids[-1]
        
        if next_oldest_peer_with_kids is not None: 
            youngest_kid_of_next_oldest_peer_with_kids = \
                self.childrenof(nodeid=next_oldest_peer_with_kids)[-1]
            my_kids_id = youngest_kid_of_next_oldest_peer_with_kids + 1
        elif parent.isleaf(): my_kids_id = peers_of_parent[-1] + 1
        else: my_kids_id = self.childrenof(nodeid=parent.idnum)[-1] + 1
       
        "Updating id numbers of younger nodes."  
        for nid_ in self.nlst[::-1]:
            if nid_ > 0:
                if self.nodes[nid_].parent >= my_kids_id:
                    self.nodes[nid_].parent += 1    

        for nid_ in self.nlst[my_kids_id:][::-1]:
            self.nodes[nid_+1] = self.nodes.pop(nid_)
            self.nodes[nid_+1].idnum = nid_ + 1

        return my_kids_id


    def add_child_of(self, parentid: int, update_attributes: bool = True, randomize_positions = True, seconds_per_node = 12) -> int:
        """
        Adds a new child node under a specified parent node and updates the tree structure.

        This method is the primary way to modify the structure of the game tree. It creates a new node, 
        assigns it an id and label, and adds it to the parent node's list of child nodes. The method also 
        updates the node's siblings' probabilities and the relevant attributes of the game tree (node list, 
        order list, and levels). Depending on the setting of update_attributes, it can also recompute the 
        positions of all nodes and the time spent on each node.

        Arguments:
            • parentid: int; The id number of the parent node under which the new node is to be added.
            • update_attributes: bool; Whether or not to update the positional and time attributes of 
                the nodes in the tree. If False, these attributes will not be updated, which can save 
                computation time when adding multiple nodes in quick succession.
            • randomize_positions: bool; If True, the node coodinates are randomized.
            • seconds_per_node: The number of seconds players will have to respond each node.

        Returns:
            • int: The id number of the newly added node.
        """
        parent = self.nodes[parentid]
        new_id = self._updateids(parentid=parentid)
        label = parent.label + abcs[len(self.childrenof(nodeid=parentid))]
        new_node = Node(nodeid=new_id, parentid=parentid, label=label, 
            level=parent.level + 1, nplayers_node=self.nplayers, root=self)
        new_node._root = self

        parent.options.append(new_node)
        self.nodes[new_id] = self.nodes[label] = new_node

        siblings = self.siblingsof(nodeid=new_id, depth=1)
        if len(siblings) > 1: self.nodes[new_id].probability[0] = round(1 / (len(siblings) - 1), 3)
        probabilities = [self.nodes[sibling].probability[0] for sibling in siblings]
        for sibling, prob in zip(siblings, probabilities):
            self.nodes[sibling].probability[0] = round(prob / sum(probabilities), 3)

        self.nlst = sorted([node for node in self.nodes if isinstance(node, int)])
        self.olst = [self.nodes[node].label for node in self.nlst]
        self.levels = [self.nodes[node].level for node in self.nlst]

        if update_attributes:
            position_dict = self.node_coordinates(screen_width=1.0, screen_height=1.0, 
                                                  randomize_visual_permutation=randomize_positions)
            for nodeid in position_dict:
                self.nodes[nodeid].positionxy = position_dict[nodeid]
            self.seconds_on_nodes(seconds_per_node=seconds_per_node, round_start_time=0, 
                seconds_per_descendant=1, buffer_between_levels=0)

        return new_id


    def _edges_to_tree(self, edges: list[int] | None, randomize_positions = True, seconds_per_node = 12) -> None:
        """
        Creates a tree from a list of edges using the add_child_of method.

        This method is a helper function used during the initialization of a game tree. It reads a list of edges, 
        which represent parent-child relationships between nodes, and uses these to create the nodes of the tree.

        Arguments:
            • edges: list[int] | None; A list of parent node ids, indexed 
                by child node ids. If None, no nodes are added to the tree.
            • randomize_positions: bool; If True, the node coodinates are randomized.
            • seconds_per_node: The number of seconds players will have to respond each node.

        Example:
            • tree = Tree(title="My_Game_Tree", edges=[0, 0, 0, 1, 1, 2, 2, 3, 3])
                This means that node 1, 2, and 3 are children of node 0, node 4 and 5 are children 
                of node 1, node 6 and 7 are children of node 2, node 8 and 9 are children of node 3.
        """
        if edges is not None:
            for child, parent in enumerate(edges):
                self.add_child_of(parentid=parent, update_attributes=child>=len(edges)-1, 
                                  randomize_positions=randomize_positions, seconds_per_node=seconds_per_node)


    @classmethod
    def random_tree_structure(self, midpoint: int | float = 10, slope: int | float = 1, allow_only_children: bool = True) -> list[int]:
        """
        Generates a random list of edges for initializing a tree structure.

        This method creates a list of edges, representing parent-child relationships between nodes, using a random 
        process controlled by the provided parameters. This allows for the generation of random game tree structures.

        Arguments:
            • midpoint: int | float; The midpoint of the sigmoid distribution to determine when to stop creating new nodes.
            • slope: int | float; The slope of the sigmoid distribution used to determine when to stop creating new nodes.
            • allow_only_children: bool; If set to True, the method can create nodes that have no siblings, allowing  
                for the creation of "chains" of nodes. If False, each parent node will have at least two children.

        Example:
            • tree = Tree(title="Random_Tree", edges=Tree.random_tree_structure(16, 0.1, True))                

        Returns:
            • list[int]: A list of parent node ids, indexed by child node ids.
        """
        import Distributions as dst

        def create_another_node(last_node_index: int, midpoint: int | float = midpoint, slope: int | float = slope):

            p_continue = dst.prob_next_round(current_round_number=last_node_index, 
                    prob_slope=slope, prob_midpoint=midpoint, constant=5.5452)
            
            return True if random.random() < p_continue else False

        edges = [0]

        while create_another_node(last_node_index=len(edges)):

            if not allow_only_children and len(edges) > 1:
                if edges[-1] != edges[-2]:
                    new_sibling = int(edges[-1])
                    edges.append(new_sibling)

            available_parents = list(range(edges[-1], len(edges)))
            edges.append(random.choice(available_parents))

        return edges if slope != 1.0 else edges[:midpoint]


    def depth_stats(self, random_walk: bool = False) -> dict[str: float, str: list[int]]:
        """
        Characterizes the distribution over the depth ('level') of leaf nodes, including:
            • mean number of steps from the root node to a leaf node
            • variance in number of these steps from root to leaf
            • number of steps for all leave nodes in an array

        Arguments:
            • random_walk: bool (optional, default False); If True, returns the 
                mean number of steps from the root node until reaching a leaf node

        Returns:
            • dict[str: float, str: list[int]]: Dictionary of depth statistics
        """
        if random_walk:
            def random_walker(parent: Node, ancestor_probs: float = None, depths: list = None):
                if depths is None: depths = []
                if ancestor_probs is None: ancestor_probs = parent.probability[0]

                for child in parent.options:
                    
                    if not child.isleaf():
                        random_walker(parent=child, ancestor_probs=ancestor_probs * child.probability[0], depths=depths)
                    else: depths.append((child.level, round(ancestor_probs * child.probability[0], child.level + 3)))

                return depths
            
            leaf_depths = random_walker(parent=self, ancestor_probs = None, depths=None)
            leaf_depths = sorted(leaf_depths, key=lambda pair: pair[0])

            mean = round(np.average(a=[leaf_level_and_probability[0] for leaf_level_and_probability in leaf_depths], 
                        weights=[leaf_level_and_probability[1] for leaf_level_and_probability in leaf_depths]), 3)
            variance = round(np.average(a=[(leaf_level_and_probability[0]-mean)**2 for leaf_level_and_probability in leaf_depths], 
                            weights=[leaf_level_and_probability[1] for leaf_level_and_probability in leaf_depths]), 3)
            
        else:
            leaf_depths = []
            for nodeid in self.nlst:
                this_node = self.nodes[nodeid]
                if this_node.isleaf():
                    leaf_depths.append(this_node.level)

            mean = round(np.average(leaf_depths), 3)
            variance = round(np.var(leaf_depths), 3)

        return {"mean": mean, "variance": variance, "leaf_depths": leaf_depths}


    def strategy_profile_maker(self) -> list[float]:
        """
        Produces strategy profiles for all players.  Invoked by __getattr__(name='strategy_profiles').

        Each strategy profile is a list of choice probabilities for all nodes in the tree.
        These are the true probabilities for the chooser and the believed probabilities for
        the predictor.  Thus, this provides the probabilities of all choices and predictions.
        
        example_tree.strategy_profiles = [
            [0.000000, 0.922963, 0.077037, 0.119203, 0.880797, 0.500000, 0.500000, 0.993307, 0.006693],
            [0.000000, 0.922963, 0.077037, 0.500000, 0.500000, 0.500000, 0.500000, 0.993307, 0.006693]
        ]"""

        strategy_profiles = []
        for player_index in range(self.nplayers):
            player_strategy_profile = [0.0 for node in self.nlst]
            for nodeid in self.nlst[::-1]:
                this_node = self.nodes[nodeid]
                if not this_node.isleaf():
                    children = self.childrenof(nodeid=nodeid)
                    if this_node.chooser[player_index]:
                        "If this player is a chooser..."
                        choice_probabilities = this_node.choice_probabilities(belief_key_prefix="")
                    else: choice_probabilities = this_node.choice_probabilities(belief_key_prefix=f"{player_index}b:")

                    for childid, probability in zip(children, choice_probabilities):
                        player_strategy_profile[childid] = probability

            strategy_profiles.append(player_strategy_profile)

        return strategy_profiles


    def _calculate_difference(self, values: list[float], as_ratio: bool, round_digs: int = False) -> float:
        """
        Calculate difference or ratio of two values.

        Arguments:
            • values: list[float]; list of two float values to compare.
            • as_ratio: bool; whether to normalize the output between -1 and 1.
            • round_digs: int (optional); number of digits to round the result to.

        Returns: 
            • float: difference or ratio of input values.
        """
        
        direction = 1 if values[0] > values[-1] else -1
        
        if as_ratio:
            "normalize vals in values to be between 0 and 1 if any value is negative"
            if any(val < 0 for val in values):
                min_val = abs(min(values))
                values = [val + min_val for val in values]

            "if values are equal, ratio is 0"
            if values[0] == values[-1]: return 0.0

            ratio = values[0] / (values[0] + values[-1])
            result = 2 * (ratio - 0.5) * direction

        else: result = values[0] - values[-1]
        
        return round(result, round_digs) if round_digs else result


    def agency(self, objective_payoffs: bool = True, as_ratio: bool = False, player_num: int = None) -> list[float] | float:
        """
        Calculate and return the degree of control that each player has over their circumstances in the game tree. 

        Control is defined by the difference in expected payoffs between the current 
        tree and a version of the tree where the player is no longer able to make choices.

        Arguments:
            • objective_payoffs: bool (optional); Determines if expected payoffs (True) or expected utilities (False) are compared.
            • as_ratio: bool (optional); Determines if output should be normalized to a value between -1 and 1. Default is False.
            • player_num: int (optional); Player number to calculate agency for. If not provided, calculates agency for all players.
                - If player_num == 'average', returns the average agency of all players.

        Returns:
            • list[float] | float: The agency values for each player in the game tree. If a specific 
                player number is provided, returns a single float for that player instead of a list.

        Raises:
            • Exception: If called on leaf node         
        """

        if self.isleaf():
            raise Exception(f"agency called on leaf node {self.idunm}!")

        def get_choosers(player_num: int) -> dict[int, list[bool]]:
            """
            Get current chooser status for each node in the game tree, and 
            an alternative where the specified player cannot make choices.

            Arguments:
                • player_num: player number to remove choice ability from.

            Returns: 
                • tuple of two dicts: current chooser status for each node, 
                    and alternative chooser status where player cannot make choices.
            """
            choosers_with_control = {}
            choosers_without_control = {}
            for node_id in self.nlst:
                if not self.nodes[node_id].isleaf():
                    choosers_with_control[node_id] = list(self.nodes[node_id].chooser)
                    choosers = list(self.nodes[node_id].chooser)
                    choosers[player_num] = False
                    choosers_without_control[node_id] = choosers

            return choosers_with_control, choosers_without_control

        def calculate_player_agency(player_num: int) -> float:
            """
            Calculate agency for a single player in the game tree.
            
            Arguments:
                • player_num: player number to calculate agency for.

            Returns: 
                • float: agency value for the specified player.
            """

            if not (0 <= player_num < self.nplayers):
                raise ValueError(f"player {player_num} not in {self.title} tree.")
            
            choosers_with_control, choosers_without_control = get_choosers(player_num)

            belief_key_prefix = f"{player_num}b:"

            if objective_payoffs: 
                expected_value_with_control = self.nodes[0].expected_payoffs(belief_key_prefix=belief_key_prefix, objective_probabilities=False, maximize_utility=True)[player_num]
            else: expected_value_with_control = self.utility(player_number=player_num, belief_key_prefix=belief_key_prefix)

            self.assign_choosers(choosers=choosers_without_control)

            if objective_payoffs: 
                expected_value_without_control = self.nodes[0].expected_payoffs(belief_key_prefix=belief_key_prefix, objective_probabilities=True)[player_num]
            else: expected_value_without_control = self.utility(player_number=player_num, belief_key_prefix=belief_key_prefix)

            self.assign_choosers(choosers=choosers_with_control)

            return self._calculate_difference(values=[expected_value_with_control, expected_value_without_control], as_ratio=as_ratio, round_digs=6)

        if player_num is None:
            return [calculate_player_agency(player_num=plr_idx) for plr_idx in range(self.nplayers)]
        elif player_num == "average":
            return round(np.average([calculate_player_agency(player_num=plr_idx) for plr_idx in range(self.nplayers)]), 6)
        else: return calculate_player_agency(player_num=player_num)


    def truthfulness(self, objective_payoffs: bool = True, as_ratio: bool = False, player_num: int = None) -> list[float] | float:
        """
        Computes the degree of truthfulness of the game tree based on the divergence between player beliefs and actual node attributes.
        
        In the context of the game tree, players' beliefs about the node attributes can differ from the actual node attributes.  This 
        function measures the degree of 'truthfulness' based on the difference between the expected payoff/utility from the perceived 
        and actual attributes of the nodes. This difference is a measure of the impact of the belief divergence on what a player values.
        
        Note: 'Truthfulness' is a bit counterintuitive in the sense that higher values indicate advantageous false beliefs, while lower 
        values indicate disadvantageous false beliefs. The maximum truthfulness is zero, which denotes no divergence between perceived 
        and actual node attributes. For a normalized measure between 0 (minimum truthfulness) and 1 (maximum truthfulness), you can use: 
        abs(tree.truthfulness(as_ratio=True)) - 1.
        
        Arguments:
            • objective_payoffs: bool; Determines if expected payoffs (True) or expected utilities (False) are compared.
            • as_ratio: bool; Determines if output should be normalized to a value between -1 and 1. Default is False.
            • player_num: int; Player number to calculate truthfulness for. If None, calculates for all players.
                - If player_num == 'average', returns the average truthfulness of all players.

        Returns:
            • list[float] | float: The agency values for each player in the game tree. If a specific 
                player number is provided, returns a single float for that player instead of a list.

        Raises:
            • Exception: If called on leaf node       
        """
        if self.isleaf():
            raise Exception(f"truthfulness called on leaf node {self.idunm}!")

        def calculate_player_truthfulness(player_num: int, objective_payoffs: bool = objective_payoffs, as_ratio: bool = as_ratio) -> float:
            """Calculates truthfulness for each player and resets the 'perspective_of_player' attribute"""

            "Saving the starting perspective so that it can be reset later"
            origional_perspective = copy.copy(self.perspective_of_player)

            "Making the tree subjective"
            self.perspective_of_player = f"@n0~p{player_num}b:"
            choice_probabilities_subjective = self.choice_probabilities(belief_key_prefix="", maximize_utility=False)
            best_option_subjective = self.options[choice_probabilities_subjective.index(max(choice_probabilities_subjective))]
            
            "Making the tree objective"
            self.perspective_of_player = None
            choice_probabilities_objective = self.choice_probabilities(belief_key_prefix="", maximize_utility=False)
            best_option_objective = self.options[choice_probabilities_objective.index(max(choice_probabilities_objective))]
            
            if objective_payoffs:
                expected_value_subjective = best_option_subjective.expected_payoffs(belief_key_prefix="", objective_probabilities=False, maximize_utility=False)[player_num]
                expected_value_objective = best_option_objective.expected_payoffs(belief_key_prefix="", objective_probabilities=False, maximize_utility=False)[player_num]
            else:
                expected_value_subjective = best_option_subjective.utility(player_number=player_num, belief_key_prefix="")
                expected_value_objective = best_option_objective.utility(player_number=player_num, belief_key_prefix="")

            "Resetting perspective attribute"
            self.perspective_of_player = origional_perspective

            return self._calculate_difference(values=[expected_value_subjective/2, expected_value_objective/2], as_ratio=as_ratio, round_digs=6)

        if player_num is None:
            return [calculate_player_truthfulness(player_num=plr_idx) for plr_idx in range(self.nplayers)]
        elif player_num == "average":
            return round(np.average([calculate_player_truthfulness(player_num=plr_idx) for plr_idx in range(self.nplayers)]), 6)
        else: return calculate_player_truthfulness(player_num=player_num)


    def alignment(self, objective_payoffs: bool = True) -> float:
        """
        Computes the degree of alignment between player interests in the game tree.
        
        This function assesses the degree to which the players' interests align or conflict within the game tree. If the interests are 
        perfectly aligned, all players would opt for the same choices, yielding the maximum collective payoff. Conversely, if interests 
        are conflicting, the choices that optimize an individual player's payoff can be detrimental to others. The degree of alignment 
        is quantified based on the angle between the joint payoff maximizing vector and a vector indicating perfect alignment.
        
        Arguments:
            • objective_payoffs: bool; Determines if expected payoffs (True) or expected utilities (False) are compared.

        Returns:
            • float: A value between -1 and 1 that represents the degree of alignment of interests between players. A value of  
                -1 indicates perfect alignment (players have the same interests), while a value of 1 indicates perfect conflict 
                (players have diametrically opposed interests). Intermediate values indicate varying degrees of alignment/conflict.

        Raises:
            • Exception: If called on leaf node    
        """

        if self.isleaf():
            raise Exception(f"alignment called on leaf node {self.idunm}!")
        
        if len(self.options) == 1: return 0.0

        "Collect expected values for each option"
        expected_values = []
        for child in self.options:
            if objective_payoffs:
                expected_values.append(child.expected_payoffs(belief_key_prefix="", objective_probabilities=True))
            else: expected_values.append([child.utility(player_number=player_idx, belief_key_prefix="") for player_idx in range(self.nplayers)])

        "Convert to a 2D numpy array for easier manipulation"
        expected_values = np.array(expected_values)

        "If all payoffs are equal or have equal sums then this is a zero sum game so return 0.0"
        if all(tuple(expected_val) == tuple(expected_values[0]) for expected_val in expected_values[1:]) or \
            all(sum(expected_val) == sum(expected_values[0]) for expected_val in expected_values[1:]): return 0.0
     
        "Find the joint payoff maximizing vector"
        joint_payoff_max = expected_values[np.argmax(np.sum(expected_values, axis=1))]
        
        "Calculate the average payoff vector"
        avg_payoff = np.mean(expected_values, axis=0)
      
        "Calculate vector from average payoff to joint payoff maximizing vector"
        alignment_vector = joint_payoff_max - avg_payoff

        "Calculate angle of this vector with respect to 'perfectly aligned' vector (45 degrees)"
        perfectly_aligned_vector = np.array([1 for plr in range(self.nplayers)]) / np.sqrt(2)  # This vector has a 45-degree angle
        dot_product = np.dot(alignment_vector, perfectly_aligned_vector)
        norms_product = np.linalg.norm(alignment_vector) * np.linalg.norm(perfectly_aligned_vector)
        cos_theta = dot_product / norms_product
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

        "Convert theta from radians to degrees and normalize it to the range [-1, 1]"
        theta_deg = np.degrees(theta)
        alignment_degree = round((90 - theta_deg) / 45 - 1, 6)
  
        return alignment_degree

    @staticmethod
    def structure_from_coordinates(branching_factor: float, asymmetry_factor: float, n_nodes: int, allow_only_children: bool = True):
        """
        Generates a tree structure based on branching factor, asymmetry factor, and a fixed number of nodes.

        Arguments:
            • branching_factor: Average number of children per node. Must be a positive integer or float.
            • asymmetry_factor: Factor controlling the asymmetry around the branching factor. 
                Ranges from 0 (perfect symmetry) to 1 (maximum asymmetry). Must be a float between 0 and 1.
            • n_nodes: Fixed number of nodes in the tree. Must be a positive integer, at least 3.
            • allow_only_children: If set to True, the method can create nodes that have no siblings, allowing for 
                the creation of "chains" of nodes. If False, each parent node will have at least two children.

        Returns:
            • List representing tree structure.
        """
        if not all(isinstance(arg, (int, float)) and arg >= 0 for arg in [branching_factor, asymmetry_factor, n_nodes]):
            raise ValueError("All inputs must be positive integers or floats.")
        if not isinstance(n_nodes, int) or n_nodes < 2:
            raise ValueError("n_nodes must be a positive integer, at least 3.")
        if not 0 <= asymmetry_factor <= 1:
            raise ValueError("asymmetry_factor must be a float between 0 and 1.")

        # n_nodes += 1
        edges, current_nodes = [0], [0]

        while len(edges) < n_nodes:
            new_nodes = []
            for node in current_nodes:
                "Determine the number of children for the current node"
                max_children = branching_factor + (branching_factor * asymmetry_factor)
                num_children = max(0, int(random.uniform(branching_factor - (branching_factor * asymmetry_factor), max_children)))
                num_children = min(num_children, n_nodes - len(edges))  # Ensure n_nodes is not exceeded

                "Add children to the tree"
                if not allow_only_children and num_children < 2 and len(edges) + 2 <= n_nodes:
                    num_children = 2  # Ensuring at least two children if allow_only_children is False
                edges.extend([node] * num_children)
                new_nodes.extend([len(edges) - idx - 1 for idx in range(num_children)])

            "If no new nodes were added and the tree size is less than n_nodes, add a child to the last node"
            if not new_nodes and len(edges) < n_nodes:
                edges.append(edges[-1])
                continue

            current_nodes = new_nodes

        return sorted(edges[1:])


    @staticmethod
    def find_vector_with_cosine_similarity(reference_vector: list[float], cosine_similarity: float, rnd_digs: int = 4):
        """
        Generate a vector with a specified cosine similarity to a given reference vector.
        
        Arguments:
            • reference_vector; list[float]: The reference vector.
            • cosine_similarity; float: The desired cosine similarity.
            • rnd_digs: int; Number of decimal places for rounding the components.
        
        Returns:
            • list[float]: A vector with the specified cosine similarity to the reference.
        """
        if not (1 < len(reference_vector) < 6):
            raise ValueError(f'reference vector must have a length between 2 and 5, not {len(reference_vector)}.') 
        
        "Normalize the reference vector"
        reference_vector = np.array(reference_vector)
        reference_norm = np.linalg.norm(reference_vector)
        reference_vector = reference_vector / reference_norm

        "The angle between the reference vector and the desired vector"
        angle = np.arccos(cosine_similarity)

        "Create a vector orthogonal to the reference vector"
        orthogonal = np.array([-reference_vector[1], reference_vector[0]] + [0] * (len(reference_vector) - 2))
        orthogonal /= np.linalg.norm(orthogonal)

        "Compute a vector in the plane spanned by 'reference' and"
        "'orthogonal' that has the desired angle with 'reference'"
        vector = np.cos(angle) * reference_vector + np.sin(angle) * orthogonal
        vector *= reference_norm  # Scale to the magnitude of the reference

        "Round and return the vector"
        return np.round(vector, rnd_digs).tolist()


    @staticmethod
    def payoffs_from_coordinates(agency: float, truthfulness: float, alignment: float, stakes: float = 0, disparity: float = 0, n_players: int = 2, n_options: int = 2, rnd_digs: int = 0) -> dict:
        """
        Generate actual and believed payoffs for a binary dictator game given agency, truthfulness, alignment, stakes, and disparity.

        Arguments:
            • agency: float; Euclidean distance representing a player's ability to affect payoff outcomes.
            • truthfulness: float; The cosine of the angle representing the player's belief accuracy.
            • alignment: float; The cosine of the angle representing the alignment of interests between players.
            • stakes: float; The base value to be added to each payoff, representing the stakes of the game.
            • disparity: float; The difference in payoff between self and other within one option.
            • n_players: int; The number of players on the tree.
            • n_options: int; The number of options on the tree.
            • rnd_digs: int; Digits to round the result to.

        Returns:
            • dict: A dictionary containing actual and believed payoffs for each option and each player.
        """
        def round_int(val: float, rnd_digs=rnd_digs):
            return round(val, rnd_digs) if rnd_digs > 0 else int(round(val, rnd_digs))

        input_types = [
            {'name': 'agency',       'input': agency,       'type': (int, float), 'min_max': None},
            {'name': 'truthfulness', 'input': truthfulness, 'type': (int, float), 'min_max': (0, 1)},
            {'name': 'alignment',    'input': alignment,    'type': (int, float), 'min_max': (0, 1)},
            {'name': 'stakes',       'input': stakes,       'type': (int, float), 'min_max': None},
            {'name': 'disparity',    'input': disparity,    'type': (int, float), 'min_max': None},
            {'name': 'n_players',    'input': n_players,    'type': int,          'min_max': None},
            {'name': 'n_options',    'input': n_options,    'type': int,          'min_max': None},
            {'name': 'rnd_digs',     'input': rnd_digs,     'type': int,          'min_max': None}
        ]

        for input_type in input_types:
            if not isinstance(input_type['input'], input_type['type']):
                raise TypeError(f"Incorrect type detected: {type(input_type['input'])} - {input_type['name']} must be an instance of {input_type['type']}!")
            if input_type['min_max']:
                if not (input_type['min_max'][0] <= input_type['input'] <= input_type['min_max'][-1]):
                    raise ValueError(f"{input_type['name']}: {input_type['input']} must be within {input_type['min_max']}!")

        "The payoff difference vector where interests are perfectly aligned"
        alignment_reference_vector = [1 * agency / np.sqrt(n_players)] * n_players

        "The objectively true payoff difference vector diverges from the alignment reference vector with a cosine similarity equal to alignment."
        payoff_difference_vector_actual = Tree.find_vector_with_cosine_similarity(reference_vector=alignment_reference_vector, cosine_similarity=alignment, rnd_digs=9)

        "The subjectively believed payoff difference vector diverges from the objectively true payoff difference vector with a cosine similarity equal to truthfulness."
        payoff_difference_vector_mental = Tree.find_vector_with_cosine_similarity(reference_vector=payoff_difference_vector_actual, cosine_similarity=truthfulness, rnd_digs=9)

        "The values used to adjust the payoffs to produce a given payoff disparity between the players"
        disparity_vector = np.linspace(disparity/2, -disparity/2, n_players)

        "Used to multiply the payoff differences so that the payoffs range between the maximum and minimum."
        option_multiples = np.linspace(1, -1, n_options)

        "Generates a dictionary of the objectively true and subjectively believed payoffs for each option."
        outcomes_actual, outcomes_mental = {}, {}
        for option in range(n_options):
            outcomes_actual[option + 1] = []
            outcomes_mental[option + 1] = []
            for player in range(n_players):
                outcomes_actual[option + 1].append(round_int(
                    stakes + disparity_vector[player] + payoff_difference_vector_actual[player] * option_multiples[option], rnd_digs
                    ))
                outcomes_mental[option + 1].append(round_int(
                    stakes + disparity_vector[player] + payoff_difference_vector_mental[player] * option_multiples[option], rnd_digs
                    ))

        "Returns a dictionary of the objectively true and subjectively believed payoffs for each option."
        return {
            'objective': outcomes_actual,
            'subjective': outcomes_mental
        }            


    @staticmethod
    def extend_payoffs_to_lower_levels(n_children: int, expected_payoffs: list[float], min_max_payoffs: list[int] = [1, 9], tolerance=0.1, max_iter=1000, rnd_digs=6):
        """
        Generates random payoffs and probabilities for child nodes in a multiplayer game,
        such that the expected payoffs for each player at the parent node remain consistent,
        and allows customization of the payoff range.

        Arguments:
            • n_children: int; Number of child nodes.
            • expected_payoffs: list[float]; Expected payoffs of the parent node for each player.
            • min_max_payoffs: list[int]; Minimum and maximum payoff range.
            • tolerance: float; Tolerance for the difference between the calculated and expected payoffs.
            • max_iter: int; Maximum number of iterations to attempt to find a solution.

        Returns:
            • Tuple of two lists; First list for payoffs (each a tuple for players) and second list for probabilities.
        """
        min_payoff, max_payoff = min_max_payoffs
        n_players = len(expected_payoffs)

        for epayoff in expected_payoffs:
            if epayoff < min_payoff:
                raise ValueError(f"Expected payoff {epayoff} is less than the minimum payoff of {min_payoff}.")
            if epayoff > max_payoff:
                raise ValueError(f"Expected payoff {epayoff} is greater than the maximum payoff of {max_payoff}.")

        for _ in range(max_iter):
            child_payoffs = []
            probabilities = []

            for _ in range(n_children - 1):
                payoff = [random.randint(min_payoff, max_payoff) for _ in range(n_players)]
                max_prob = min(1, min(abs(remaining_payoff / p) if p != 0 else 1 for remaining_payoff, p in zip(expected_payoffs, payoff)))
                prob = random.uniform(0, max_prob)
                child_payoffs.append(tuple(payoff))
                probabilities.append(prob)

            "Adjust the last child's payoff and probability"
            last_prob = round(1 - sum(probabilities), rnd_digs)
            if last_prob <= 0:
                continue

            last_payoffs = []
            for player, expected_payoff in enumerate(expected_payoffs):
                remaining_payoff = expected_payoff - sum(payoff[player] * prob for payoff, prob in zip(child_payoffs, probabilities))
                last_payoff = remaining_payoff / last_prob if last_prob > 0 else random.randint(min_payoff, max_payoff)
                last_payoffs.append(max(min_payoff, min(max_payoff, round(last_payoff))))

            child_payoffs.append(tuple(last_payoffs))
            probabilities.append(last_prob)

            "Verification"
            calculated_payoffs = [0] * n_players
            for child, prob in zip(child_payoffs, probabilities):
                for idx, payoff in enumerate(child):
                    calculated_payoffs[idx] += payoff * prob

            if all(abs(calculated_payoff - expected_payoff) <= tolerance for calculated_payoff, expected_payoff in zip(calculated_payoffs, expected_payoffs)):
                return child_payoffs, [round(probability, rnd_digs) for probability in probabilities]

        "Default answer if a solution is not found"
        default_payoff = [round(expected_payoff / n_children) for expected_payoff in expected_payoffs]
        default_probabilities = [round(1/n_children, rnd_digs)] * n_children
        return [tuple(default_payoff) for _ in range(n_children)], default_probabilities


    @staticmethod
    def tree_from_coordinates(agency: float, truthfulness: float, alignment: float, stakes: float = 0, disparity: float = 0, 
                            branching_factor: float = 2, asymmetry_factor: float = 0, n_nodes: int = 3, n_players: int = 2, 
                            allow_only_children: bool = True, min_max_payoffs: list[int] = [-9, 9], rnd_digs: int = 4) -> 'Tree':
        """
        Constructs a game tree from specified parameters, representing various psychological and game-theoretic dimensions.

        The tree is built based on a set of input parameters that define the structure and payoffs at each node.
        This function allows for creating complex game trees suited for experiments in behavioral game theory.

        Arguments:
            • agency: float; Represents a player's ability to affect outcomes.
            • truthfulness: float; Degree of accuracy in a player's belief about payoffs.
            • alignment: float; Represents the alignment of interests between players.
            • stakes: float; Base value for each payoff, indicating the game's stakes.
            • disparity: float; Difference in payoffs between self and other within an option.
            • branching_factor: float; Average number of children per node in the tree.
            • asymmetry_factor: float 0 - 1; Degree of asymmetry in the tree's branching. 
            • n_nodes: int; Total number of nodes in the tree.
            • n_players: int; Number of players on the tree.
            • allow_only_children: bool; If True, allows nodes to have only one child.
            • min_max_payoffs: list[int]; Range of possible payoffs for players.
            • rnd_digs: int; Number of decimal places for rounding.

        Returns:
            • Tree: An instance of a Tree object with the specified parameters.
        """
        "Generate the structure of the tree based on branching factor and asymmetry."
        edges = Tree.structure_from_coordinates(branching_factor=branching_factor, asymmetry_factor=\
                                        asymmetry_factor, n_nodes=n_nodes, allow_only_children=allow_only_children)
        
        "Create the tree with a title based on the parameters."
        title = f"Tree-{round(agency,2)}-{round(truthfulness,2)}-{round(alignment,2)}-{round(stakes,2)}-{round(disparity,2)}"
        title += f"-{round(branching_factor,2)}-{round(asymmetry_factor,2)}-{n_nodes}-{n_players}"
        tree = Tree(title=title, nplayers=n_players, edges=edges)

        "Generate payoffs for the children of the root node"
        n_children_of_root = len([nodeid for nodeid in edges if nodeid == 0])
        payoffs_under_root = Tree.payoffs_from_coordinates(agency=agency, truthfulness=truthfulness, alignment=alignment, 
                                            stakes=stakes, disparity=disparity, n_players=n_players, 
                                            n_options=n_children_of_root, rnd_digs=0)

        #TODO make this work for believed payoffs, not just the objective payoffs.
        tree.assign_payoffs(payoffs=payoffs_under_root.get('objective')) 
        tree.assign_choosers(choosers="random")

        "Extend payoffs to lower levels of the tree"
        individual_levels = sorted(list(set(tree.levels)))
        for ilevel in individual_levels[1:]:
            for nodeid, level in zip(tree.nlst, tree.levels):
                if level == ilevel:
                    node = tree.nodes[nodeid]
                    if not node.isleaf():
                        "Generate and assign payoffs and probabilities for child nodes."
                        payoffs, probabilities = Tree.extend_payoffs_to_lower_levels(
                            n_children=len(node.options), expected_payoffs=node.payoffs, 
                            min_max_payoffs=min_max_payoffs, rnd_digs=rnd_digs)
                        for idx, child in enumerate(node.options):
                            tree.assign_payoffs(payoffs={child.idnum: payoffs[idx]})
                            tree.assign_probabilities(probabilities={child.idnum: probabilities[idx]})
                        "Erasing the payoffs from intermediate nodes once they have been extended to their children."
                        tree.assign_payoffs(payoffs={node.idnum: [0] * n_players})
                            
        return tree

