"""
=============================================================================================================
======== Analysis Script for Higher-order Theory of Mind Experiment by Gregory N. Stanley 12/12/2024 ========
=============================================================================================================
"""
import GameTree as gt, plotly.graph_objects as go, itertools as it, pandas as pd, numpy as np
from scipy.stats import chi2_contingency, friedmanchisquare, wilcoxon
import random, pprint, math, json, copy, ast, os
pp = pprint.PrettyPrinter(indent=2)
from scipy.special import softmax 
from pathlib import Path
dark_mode = False

"Project paths (relative to this file)"
ROOT = Path(__file__).resolve().parent
file_path_raw     = ROOT / "Data" / "Raw"
file_path_clean   = ROOT / "Data" / "Clean"
file_path_figures = ROOT / "Figures"

"Ensure directories exist"
for file_path in [file_path_raw, file_path_clean, file_path_figures]:
    file_path.mkdir(exist_ok=True)

"Default layout for Plotly figures."
txt_color, txtfam = "white" if dark_mode else "black", "Calibri"
fig_lay = {
    "template": "plotly_dark" if dark_mode else "plotly_white",
    "font": dict(family=txtfam, color=txt_color, size=24),
    "tickfont": dict(family=txtfam, color=txt_color, size=30),
    "titlefont_size": 48, "title_x": 0.5, "title_y": 0.96, "scale": ("x", 1),
    "colorscales": ['Viridis', 'Plasma', 'Inferno', 'matter', 'haline', 'thermal', 'dense', 'Magma'],
    "annotations": {"font":  dict(family=txtfam, color=txt_color, size=34), "showarrow": False}, 
    "xaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34), 
        "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "yaxis" : {"title_font": dict(family=txtfam, color=txt_color, size=34), 
        "tickfont": dict(size=30, family=txtfam, color=txt_color)},
    "hoverlabel": dict(font_size=30, font_family=txtfam)
}

"32 Payoff structures used in this experiment:"
selected_trees = [
    (2, 1, 4, 3, 1, 3, 2, 4),
    (2, 1, 4, 3, 3, 2, 1, 4),
    (2, 1, 4, 3, 4, 2, 1, 3),
    (3, 4, 2, 1, 2, 3, 4, 1),
    (2, 1, 3, 4, 2, 3, 4, 1),
    (3, 4, 1, 2, 1, 3, 2, 4),
    (3, 4, 1, 2, 3, 2, 1, 4),
    (3, 4, 1, 2, 4, 2, 1, 3),
    (2, 1, 4, 3, 2, 3, 4, 1),
    (3, 1, 4, 2, 2, 3, 4, 1),
    (3, 2, 4, 1, 3, 2, 4, 1),
    (3, 4, 2, 1, 3, 2, 1, 4),
    (2, 1, 3, 4, 3, 2, 1, 4),
    (2, 3, 1, 4, 3, 2, 4, 1),
    (2, 4, 1, 3, 2, 3, 4, 1),
    (3, 4, 1, 2, 2, 3, 4, 1),
    (2, 1, 3, 4, 1, 3, 4, 2),
    (2, 1, 3, 4, 3, 2, 4, 1),
    (2, 1, 3, 4, 4, 2, 3, 1),
    (3, 4, 1, 2, 2, 3, 1, 4),
    (2, 1, 4, 3, 2, 3, 1, 4),
    (3, 4, 2, 1, 1, 3, 4, 2),
    (3, 4, 2, 1, 3, 2, 4, 1),
    (3, 4, 2, 1, 4, 2, 3, 1),
    (2, 1, 3, 4, 2, 3, 1, 4),
    (3, 1, 2, 4, 2, 3, 1, 4),
    (3, 2, 1, 4, 3, 2, 1, 4),
    (3, 4, 1, 2, 3, 2, 4, 1),
    (2, 1, 4, 3, 3, 2, 4, 1),
    (2, 3, 4, 1, 3, 2, 1, 4),
    (2, 4, 3, 1, 2, 3, 1, 4),
    (3, 4, 2, 1, 2, 3, 1, 4)
]

"Hedden and Zhang's (2002) Payoff structures, included for comparison."
juns_games = [
    (3,	4, 1, 2, 2, 3, 4, 1),
    (3,	4, 1, 2, 3, 2, 1, 4),
    (3,	4, 2, 1, 3, 2, 1, 4),
    (3,	2, 1, 4, 4, 2, 1, 3),
    (3,	1, 2, 4, 1, 3, 4, 2),
    (3,	4, 1, 2, 4, 2, 3, 1),
    (3,	2, 1, 4, 1, 3, 2, 4),
    (3,	1, 2, 4, 3, 2, 4, 1),
    (3,	1, 4, 2, 4, 2, 3, 1),
    (3,	4, 1, 2, 2, 3, 1, 4),
    (3,	2, 1, 4, 3, 2, 4, 1),
    (3,	1, 2, 4, 2, 3, 1, 4),
    (3,	2, 4, 1, 1, 3, 2, 4),
    (3,	4, 1, 2, 3, 2, 4, 1),
    (3,	4, 1, 2, 1, 3, 2, 4),
    (3,	1, 2, 4, 4, 2, 1, 3),
    (3,	2, 1, 4, 2, 3, 4, 1),
    (3,	4, 1, 2, 4, 2, 1, 3),
    (3,	4, 2, 1, 1, 3, 4, 2),
    (3,	4, 1, 2, 1, 3, 4, 2),
    (2,	4, 1, 3, 4, 2, 1, 3),
    (2,	1, 3, 4, 1, 3, 4, 2),
    (2,	3, 1, 4, 2, 3, 4, 1),
    (2,	1, 4, 3, 4, 2, 1, 3),
    (2,	1, 3, 4, 3, 2, 1, 4),
    (2,	3, 1, 4, 1, 3, 4, 2),
    (2,	1, 3, 4, 2, 3, 4, 1),
    (2,	3, 1, 4, 4, 2, 1, 3),
    (2,	1, 4, 3, 2, 3, 1, 4),
    (2,	1, 3, 4, 1, 3, 2, 4),
    (2,	1, 3, 4, 4, 2, 3, 1),
    (2,	4, 1, 3, 3, 2, 4, 1),
    (2,	1, 4, 3, 1, 3, 2, 4),
    (2,	1, 3, 4, 4, 2, 1, 3),
    (2,	4, 1, 3, 2, 3, 1, 4),
    (2,	1, 3, 4, 3, 2, 4, 1),
    (2,	1, 3, 4, 2, 3, 1, 4),
    (2,	1, 4, 3, 3, 2, 1, 4),
    (2,	4, 1, 3, 1, 3, 2, 4),
    (2,	3, 1, 4, 4, 2, 3, 1)
]

"""
===================================================================================
================================ MAKING THE GAMES =================================
===================================================================================
"""
def stay_or_move(payoffs_plr_1: tuple[int], payoffs_plr_2: tuple[int], model_type: str, ToM_level: int) -> str:
    """
    Computes player 1's optimal choice at the root node depending on the type of model and Theory of Mind level.

    Arguments:
        • payoffs_plr_1: tuple[int]; Player 1's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_plr_2: tuple[int]; Player 2's ordinal payoffs in the 3-step Stackelberg game
        • model_type: 'BM' | 'NM'; The type of theory used to model kth-order Theory of Mind
            - 'BM': Blind Spot Model represents lower-order players as blind to game states greater than k.
            - 'NM': Near Sight Model represents lower-order players as converting choice nodes to chance nodes after k steps.
        • ToM_level: 1 | 2; Determines if the player thinks at the first- or second-order of Theory of Mind.

    Returns:
        • choice: 'move' | 'stay'; Player 1's optimal choice at the root node.
    """
    a1, b1, c1, d1 = payoffs_plr_1
    a2, b2, c2, d2 = payoffs_plr_2

    if ToM_level == 2:
        "Both models use self-interested backwards induction to find player 1 optimal choice."
        pay_plr_1_cd = max(c1, d1)
        pay_plr_2_cd = c2 if c1 > d1 else d2
        pay_plr_1_bcd = b1 if b2 > pay_plr_2_cd else pay_plr_1_cd
        return 'move' if pay_plr_1_bcd > a1 else 'stay'
    
    elif ToM_level == 1: 
        "For first-order players:"
        if model_type == 'BM':
            "For the Blind Spot Model:"
            pay_plr_1_bc = b1 if b2 > c2 else c1
            return 'move' if pay_plr_1_bc > a1 else 'stay'

        elif model_type == 'NM':
            "For the Near Sight Model:"
            pay_plr_1_cd = (c1 + d1) / 2
            pay_plr_2_cd = (c2 + d2) / 2

            if b2 > pay_plr_2_cd:
                pay_plr_1_bcd = b1
            elif b2 < pay_plr_2_cd:
                pay_plr_1_bcd = pay_plr_1_cd
            else:
                pay_plr_1_bcd = (b1 + pay_plr_1_cd) / 2 
            
            if a1 == pay_plr_1_bcd:
                return 'rand'

            return 'move' if pay_plr_1_bcd > a1 else 'stay' 

        elif model_type == 'BPH':
            "For the branch pruning heuristic theory"
            sum_b = sum([b1, b2])
            sum_c = sum([c1, c2])
            sum_d = sum([d1, d2])
            pay_plr_1_cd = (c1 + d1) / 2

            if sum_b > max(sum_c, sum_d):
                pay_plr_1_bcd = b1

            elif sum_b < max(sum_c, sum_d):
                if sum_c > sum_d:
                    pay_plr_1_bcd = c1            
                elif sum_c < sum_d:
                    pay_plr_1_bcd = d1  
                else:
                    pay_plr_1_bcd = pay_plr_1_cd  

            else:
                if sum_c > sum_d:
                    pay_plr_1_bcd = (b1 + c1) / 2           
                elif sum_c < sum_d:
                    pay_plr_1_bcd = (b1 + d1) / 2  
                else:
                    pay_plr_1_bcd = (b1 + pay_plr_1_cd) / 2                                    

            if a1 == pay_plr_1_bcd:
                return 'rand'

            return 'move' if pay_plr_1_bcd > a1 else 'stay' 

        elif model_type == 'S23':
            return 'move' if a1 < 3 else 'stay'
        
        else:
            raise ValueError(f"model_type must be a string literal: 'BJH' | 'NM'!")
    else:
        raise ValueError(f"ToM_level must be 1 or 2.")


def expected_payoffs_near_sight_model(node: gt.Node, level_k: int) -> list[int]:
    """
    Calculate expected payoffs for a node using the Near Sight Model of Theory of Mind (ToM).
    Recursively analyzes child nodes until the depth of ToM reasoning (level_k) is exhausted.

    Arguments:
        • node: gt.Node; The current game tree node.
        • level_k: int; The remaining depth of ToM reasoning.

    Returns:
        • expected_payoffs: list[int]; A list of expected payoffs for each player at this node
    """
    if node.isleaf():
        "If node is a leaf, return its payoffs."
        return list(node.payoffs)
    
    "Initialize expected payoffs from the current node's payoffs."
    parent_payoffs = list(node.payoffs)

    if level_k > 0:
        "Recursive case: calculate expected payoffs from children nodes."

        "Find expected payoffs via recursion while depleating ToM reasoning level."
        child_payoffs = [
            expected_payoffs_near_sight_model(node=child, level_k=level_k - 1) 
            for child in node.options
            ]

        "The chooser will select the child node with the highest payoff for themselves."
        chooser_index = node.chooser.index(True)
        chosen_payoffs = max(child_payoffs, key=lambda x: x[chooser_index])

        "Add these payoffs to the payoffs at the parent node, if any."
        expected_payoffs = [sum(payoffs) for payoffs in zip(parent_payoffs, chosen_payoffs)]

    else:
        "Base case: use expected value heuristic once level_k is depleted."
        expected_payoffs = parent_payoffs
        for child in node.options:
            child_probability = child.probability[0]
            for player_index, expected_payoff in enumerate(
                expected_payoffs_near_sight_model(node=child, level_k=level_k - 1)):
                expected_payoffs[player_index] += expected_payoff * child_probability

    return expected_payoffs


def expected_payoffs_blind_spot_model(node: gt.Node, level_k: int) -> list[int]:
    """
    The procees of finding the expected payoffs at a node according to the Blind Spot Model.

    Calculate expected payoffs for a node using the Blind Spot Model of Theory of Mind (ToM).
    When cognitive resources are depleted (level_k = 0), it is blind to all but the first child node. 

    Arguments:
        • node: gt.Node; The current game tree node.
        • level_k: int; The remaining depth of ToM reasoning.

    Returns:
        • expected_payoffs: list[int]; A list of expected payoffs for each player at this node
    """
    if node.isleaf():
        "If node is a leaf, return its payoffs."
        return list(node.payoffs)
    
    "Initialize expected payoffs from the parent node's payoffs."
    parent_payoffs = list(node.payoffs)

    if level_k > 0:
        "Recursive case: calculate expected payoffs from children nodes."

        "Find expected payoffs via recursion while depleating ToM reasoning level."
        child_payoffs = [
            expected_payoffs_blind_spot_model(node=child, level_k=level_k - 1) 
            for child in node.options
            ]

        "The chooser will select the child node with the highest payoff for themselves."
        chooser_index = node.chooser.index(True)
        chosen_payoffs = max(child_payoffs, key=lambda x: x[chooser_index])

        "Add these payoffs to the payoffs at the parent node, if any."
        expected_payoffs = [sum(payoffs) for payoffs in zip(parent_payoffs, chosen_payoffs)]

    else:
        "Base case: once level_k is depleted use heuristic of only attending to the payoffs of the first child node."
        first_child_payoffs = expected_payoffs_blind_spot_model(node=node.options[0], level_k=level_k - 1)
        expected_payoffs = [sum(payoffs) for payoffs in zip(parent_payoffs, first_child_payoffs)]

    return expected_payoffs


def classify_game(payoffs_plr_1: tuple[int], payoffs_plr_2: tuple[int]):
    """
    Arguments:
        • payoffs_plr_1: tuple[int]; Player 1's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_plr_2: tuple[int]; Player 2's ordinal payoffs in the 3-step Stackelberg game    
    """
    a1, b1, c1, d1 = payoffs_plr_1
    a2, b2, c2, d2 = payoffs_plr_2

    "Determine optimal plr 1 choices at root node for each model."
    BM2 = stay_or_move(payoffs_plr_1, payoffs_plr_2, 'BM', 2)
    BM1 = stay_or_move(payoffs_plr_1, payoffs_plr_2, 'BM', 1)
    NM2 = stay_or_move(payoffs_plr_1, payoffs_plr_2, 'NM', 2)
    NM1 = stay_or_move(payoffs_plr_1, payoffs_plr_2, 'NM', 1)
    BPH =  stay_or_move(payoffs_plr_1, payoffs_plr_2, 'BPH', 1)
    S23 =  stay_or_move(payoffs_plr_1, payoffs_plr_2, 'S23', 1)

    "String to summarize the BM2, BM1, NM2, and NM1"
    ms_dict = {'move': 'M', 'stay': 'S', 'rand': 'R'}
    quadruplet = "".join([ms_dict[ms] for ms in [BM2, BM1, NM2, NM1]])

    "Create title from payoffs"
    title = f"ToM+{quadruplet}-{a1}{b1}{c1}{d1}-{a2}{b2}{c2}{d2}"

    "Classify games to aid sorting and filtering."
    diagnostic_BM = BM2 != BM1
    diagnostic_NM = NM2 != NM1 
    trivial = b2 < min(c2, d2) or b2 > max(c2, d2)
    no_brainer = a1 <= 1 or a1 >= 4

    "Expresses which theories predict mistakes, where ⊥ = mistake and ⊤ = correct."
    mistake = "⊥" if diagnostic_BM else "⊤"
    mistake += "⊥" if diagnostic_NM else "⊤"

    "Records if the tree will be included in the next experiment"
    selected = tuple(payoffs_plr_1 + payoffs_plr_2) in set(selected_trees)

    "Games where both theories make the same predictions can be used to diagnose ToM level."
    concensus = diagnostic_BM and diagnostic_NM

    "Games where both theories make different predictions can be used to adjudicate theories."
    controversy = BM1 != NM1 and NM1 != 'rand'

    "Way of categorizing games based on payoff comparisons."
    updown = "↑" if a1 < b1 else "↓"
    updown += "↑-" if b1 < max(c1, d1) else "↓-"
    updown += "↑" if b2 < c2 else "↓"
    updown += "↑" if c2 < d2 else "↓"

    game = {
        'pay-a1': a1, 'pay-b1': b1, 
        'pay-c1': c1, 'pay-d1': d1,
        'pay-a2': a2, 'pay-b2': b2, 
        'pay-c2': c2, 'pay-d2': d2,
        'payoffs_plr_1': payoffs_plr_1, 
        'payoffs_plr_2': payoffs_plr_2,
        'diagnostic-BM': diagnostic_BM,
        'diagnostic-NM': diagnostic_NM,
        'BM2': BM2, 'BM1': BM1,
        'NM2': NM2, 'NM1': NM1,
        'quadruplet': quadruplet,
        'consensus': concensus,
        'controversy': controversy,
        'no-brainer': no_brainer,
        'trivial': trivial, 
        'up-down': updown,
        'BPH': BPH, 'S23': S23,
        'mistake': mistake, 
        'selected': selected
    }   

    return (title, game) 


def game_classification() -> dict[str: dict]:
    """
    Generates a dictionary of all 2-player 3-step alternating-choice games with ordinal payoffs 1, 2, 3, 
    and 4 and classifies them based on the optimal choices from each model of higher-order Theory of Mind.
    """
    games = {}

    "Iterating over all possible payoff permutations."
    payoff_permutations_individual = list(it.permutations(range(1, 5)))
    for payoffs_plr_1 in payoff_permutations_individual:
        for payoffs_plr_2 in payoff_permutations_individual:
            title, game = classify_game(payoffs_plr_1=payoffs_plr_1, payoffs_plr_2=payoffs_plr_2)
            games[title] = game

    return games


def tom_game_inventory(games: dict = game_classification(), 
                        file_path: str = file_path_clean, file_name: str = "ToM+Game_Inventory.csv") -> pd.DataFrame:
    """
    Saves the games dictionary as a CSV file after converting it to a pandas DataFrame. It first checks if the 
    CSV exists within the specified folder. If so, it returns the pre-existing CSV. Otherwise, it generates the 
    DataFrame, saves it, retrieves it, and returns it.

    Arguments:
        • games: dict; The dictionary of games to be converted and saved.
        • file_path: str; The path where the CSV file will be saved.
        • file_name: str; The name of the CSV file.

    Returns:
        • df: pd.DataFrame; The dataframe saved or loaded from the file.
    """
    if not file_name.endswith(".csv"):
        file_name += ".csv"
    full_path = os.path.join(file_path, file_name)

    if os.path.exists(full_path):
        df = pd.read_csv(full_path)
        if 'Unnamed: 0' in df.columns: 
            del df['Unnamed: 0']
    else:
        df = pd.DataFrame.from_dict(games, orient='index')
        df.index.name = 'Title'
        df.to_csv(full_path, encoding='utf-8-sig')

    return df


def generate_predetermined_trials(n_humans: int = 1, save_trials: bool = False, 
                                  save_trees: bool = False, print_: bool = False) -> None:
    """
    Creates a JSON file used to program the Morality Game to execute a predetermined sequence of trials.

    Arguments:
        • n_humans: int; Number of human participants.
        • save_trials: bool; If true, saves the JSON file.
        • save_trees: bool; If true, saves the game trees as JSON files.
        • print_: bool; If true, prints to terminal.

    Returns:
        • None; Saves a JSON file.
    """
    "Retrieving dataframe from CSV file."
    games_df = tom_game_inventory()

    "Only include games to be used in the next experiment."
    games_df = games_df[games_df['selected'] == True]
    games_df = games_df.drop('selected', axis=1)
    desired_order = ["SSSM", "SMSS", "MMMS", "MSMM", "SMSM", "SSSS", "MSMS", "MMMM"]
    games_df['quadruplet'] = pd.Categorical(games_df['quadruplet'], categories=desired_order, ordered=True)
    games_df = games_df.sort_values('quadruplet')

    rounds = []
    games_mat = games_df.to_numpy()
    for idx, game in enumerate(games_mat):
        if idx > -1 and idx <= 77:
            payoffs_plr_1 = tuple(game[1:5])
            payoffs_plr_2 = tuple(game[5:9])
            a1, b1, c1, d1 = payoffs_plr_1
            a2, b2, c2, d2 = payoffs_plr_2    
            payoffs = {"1": [a1, a2], "3": [b1, b2], "5": [c1, c2], "6": [d1, d2]}
            BM2, BM1, NM2, NM1, quadruplet = game[13:18]
            pay_plr_2_cd = c2 if c1 > d1 else d2
            choice_ab = [0, 1] if BM2 == "move" else [1, 0]
            choice_bc = [0, 1] if pay_plr_2_cd > b2 else [1, 0]
            choice_cd = [0, 1] if d1 > c1 else [1, 0]
            sprofiler = [0] + choice_ab + choice_bc + choice_cd
            sprofiles = [[0] + [0.5] * 6, sprofiler]

            tree = {
                "nplayers": 2,
                "title": game[0],
                "edges": [0, 0, 2, 2, 4, 4],
                "adjacency_matrix": [[1, 1], [1, 1]],
                "choosers": {"0": [True, False], "2": [False, True], "4": [True, False]},
                "payoffs": payoffs       
            }

            if save_trees:
                tree_ = gt.Tree(title=game[0], nplayers=2, edges=tree['edges'], randomize_positions=False)
                tree_.seconds_on_nodes(seconds_per_node=8, seconds_per_descendant=0, buffer_between_levels=0)
                tree_.assign_choosers(choosers={0: [True, False], 2: [False, True], 4: [True, False]})
                tree_.assign_payoffs(payoffs={int(key): val for key, val in payoffs.items()})
                tree_.strategy_profiles = sprofiles
                tree_.to_json("./Inputs/Trees/Json_Trees")

            around = [{
                "player_keys": [f"h{idx+1}", f"r{idx+1}"], 
                "strategy_profiles": sprofiles, "tree": tree
            } for idx in range(n_humans)]

            rounds.append(around)


    players = {
        **{f"h{idx}": None for idx in range(1, n_humans + 1)}, 
        **{f"r{idx}": None for idx in range(1, n_humans + 1)}
    }

    predetermined_trials = {
        "orders": {
            "randomize_trial_order": True
        },
        "players": players,
        "rounds": rounds
    }

    if print_:
        pp.pprint(predetermined_trials)

    if save_trials:
        file_name_predetermined = f"predetermined_trials_ToM_{n_humans}_humans.json"
        with open(os.path.join(file_path_clean, "Predetermined", file_name_predetermined), "w") as file: 
            json.dump(predetermined_trials, file, indent=4)  


def compute_reaction_time(tree: gt.Tree) -> gt.Tree:
    """
    Helper function to compute reaction times. 
    Depricated in new version of Morality Game.
    """
    fade_in_delay = 3
    round_started_time = tree.timestamps['round_started_time']
    for nodeid in tree.timeline():
        this_node = tree.nodes[nodeid]
        if nodeid == 0:
            choice = this_node.choice[0]
            if isinstance(choice, dict):
                choice = copy.deepcopy(choice)
                node_started_time = this_node.time[0] + round_started_time
                choice_timestamp = choice['timestamp']
                choice_rt = choice_timestamp - node_started_time
                down_up_diff = choice['rtimeup'] - choice['rtimedn']
                choice['rtimeup'] = max(choice_rt + down_up_diff - fade_in_delay, 0)
                choice['rtimedn'] = max(choice_rt - fade_in_delay, 0)
                this_node.choice[0] = choice
                    
        if nodeid == 2:
            prediction = this_node.prediction[0]
            if isinstance(prediction, dict):
                prediction_timestamp = prediction['timestamp']
                prediction_rt = prediction_timestamp - round_started_time
                down_up_diff = prediction['rtimeup'] - prediction['rtimedn']
                prediction['rtimeup'] = max(prediction_rt + down_up_diff - fade_in_delay, 0.0)
                prediction['rtimedn'] = max(prediction_rt - fade_in_delay, 0.0)
                this_node.prediction[0] = prediction

    return tree


def tree_lists(file_path: str = file_path_raw) -> dict[str: list[gt.Tree]]:
    "List all JSON files in the directory"
    json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]
    trees = []
    
    for json_file in json_files:
        full_path = os.path.join(file_path, json_file)
        try:
            "Directly read the JSON file without altering its name"
            with open(full_path, 'r') as file:
                trees_data = json.load(file)
                trees_data = trees_data[:-1]
            
            "Convert the list of dictionaries to Tree instances"
            trees += gt.Tree.list_of_trees(trees_data)
            for tree in trees:
                tree = compute_reaction_time(tree=tree)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    trees: list[gt.Tree]
    trees = [tree for tree in trees if tree.players[0]['player_type'] == 'participant']

    return trees


"""
===================================================================================
=============================== PROCESSING THE DATA ===============================
===================================================================================
"""
def dataframe(file_path: str = file_path_clean, 
              file_name: str = f"Morality_Game_Study_Results_Higher_ToM_Merged.csv",
              recreate_csv: bool = False) -> pd.DataFrame:
    """
    Retrieves the results CSV file or generates them from the raw data and returns them as a Pandas dataframe.

    Arguments:
        • file_path: str; File path of the experiment results CSV.
        • file_name: str; File name of the experiment results CSV.

    Returns:
        • results_df: pd.DataFrame; The experiment results.    
    """
    full_path = os.path.join(file_path, file_name)

    if not recreate_csv and os.path.exists(full_path):
        results_df = pd.read_csv(full_path)    
        if 'Unnamed: 0' in results_df.columns:
            del results_df['Unnamed: 0']
        return results_df

    results_df = gt.Tree.trees_list_to_dataframe(
        trees_with_responses=tree_lists(), expanded_format=True)
    results_df.to_csv(full_path, encoding='utf-8-sig')

    return results_df


def cleaned_df(recreate_csv: bool = False, print_: bool = False, max_abdications: int = 31) -> dict[str: dict[str: list] | pd.DataFrame]:
    """
    Preprocesses results data into a format for efficient analysis.
    Drops participants with > max_abdications abdicated root choices.
    Prints a distribution table of abdications per participant.

    Arguments:
        • recreate_csv: bool; Recreates the CSV if true.
        • print_: bool; Prints information to terminal if true.
        • max_abdications: int; Participants with > max_abdications are excluded.

    Returns:
        • dict with 'df' filtered by the exclusion, plus column dicts.
    """
    file_name = "Morality_Game_Study_Results_Higher_ToM_Analyzed.csv"
    full_path = os.path.join(file_path_clean, file_name)

    analysis_cols = {
        'payoffs_plr_1': [],
        'payoffs_plr_2': [],
        'diagnostic-BM': [],
        'diagnostic-NM': [],
        'BM2': [], 'BM1': [],
        'NM2': [], 'NM1': [],
        'quadruplet': [],
        'consensus': [],
        'controversy': [],
        'no-brainer': [],
        'trivial': [],
        'up-down': [],
        'BPH': [],
        'S23': [],
        'mistake': [],
        'selected': []
    }

    response_cols = {
        'choice': [],
        'prediction': [],
    }

    if not recreate_csv and os.path.exists(full_path):
        tom_trees_df = pd.read_csv(full_path)
        if 'Unnamed: 0' in tom_trees_df.columns:
            del tom_trees_df['Unnamed: 0']

        "Exclude participants that abdicated too many reponses"
        abd_by_participant = (
            tom_trees_df.assign(_abd=lambda d: d['choice'].astype(str).str.lower().eq('none'))
                       .groupby('participant_number')['_abd'].sum().astype(int)
        )

        "Exclude > max_abdications"
        keepers = abd_by_participant[abd_by_participant <= max_abdications].index
        excluded = abd_by_participant[abd_by_participant > max_abdications]
        if print_:
            # Pretty one-line distribution like: (0) 43 (1) 13 (2) 11 ...
            dist = abd_by_participant.value_counts().sort_index()
            print("\nN Abdications Per Participant Distribution (Root Node Choices):")
            summary = " ".join(f"({n}) {cnt}" for n, cnt in dist.items())
            print(summary)

            if len(excluded):
                print(f"\nExcluded participants for excessive abdications (> {max_abdications}):")
                for pn, n_abd in excluded.items():
                    print(f"  participant {pn}: abdications = {n_abd}")
                print("")

        tom_trees_df = tom_trees_df[tom_trees_df['participant_number'].isin(keepers)].copy()
        tom_trees_df.reset_index(drop=True, inplace=True)

        if print_:
            pd.set_option('display.max_columns', 22)
            print(tom_trees_df)
            print("")

        primary_cols_, analysis_cols_, response_cols_ = {}, {}, {}
        for key in tom_trees_df.columns:
            if key in response_cols.keys():
                response_cols_[key] = tom_trees_df[key].tolist()
            elif key in analysis_cols.keys():
                analysis_cols_[key] = tom_trees_df[key].tolist()
            else:
                primary_cols_[key] = tom_trees_df[key].tolist()

        return {
            'primary_cols': primary_cols_,
            'analysis_cols': analysis_cols_,
            'response_cols': response_cols_,
            'df': tom_trees_df,
        }

    "---------- If recreating CSV from raw data ----------"
    tom_trees_df = dataframe(file_path=file_path_clean, recreate_csv=recreate_csv)

    cols_to_drop = ['experiment_setting_keys', 'experiment_setting_values', 'room',
                    'payoffs__p0', 'payoffs__p1', 'payoffs_B_p0', 'payoffs_B_p1', 'payoffs_BB_p0', 'payoffs_BB_p1']
    for col_name in list(tom_trees_df.columns):
        for col_start in ['beliefs', 'options', 'probability', 'positionxy', 'info_set']:
            if col_name.startswith(col_start):
                cols_to_drop.append(col_name)
        if col_name.startswith('choice'):
            if col_name not in ['choice__p0', 'choice_B_p1', 'choice_BB_p0', 'choice_data__p0']:
                cols_to_drop.append(col_name)
        if col_name.startswith('prediction'):
            if col_name not in ['prediction_B_p0', 'prediction_data_B_p0']:
                cols_to_drop.append(col_name)

    tom_trees_df = tom_trees_df.drop(columns=cols_to_drop)

    "Participant numbers"
    tom_trees_df['player_uuids_str'] = tom_trees_df['player_uuids'].apply(lambda x: str(x))
    unique_uuids = sorted(set(tom_trees_df['player_uuids_str']))
    uuid_to_number = {uuid: i for i, uuid in enumerate(unique_uuids, start=1)}
    tom_trees_df.insert(loc=4, column='participant_number', value=tom_trees_df['player_uuids_str'].map(uuid_to_number))
    tom_trees_df = tom_trees_df.sort_values(by=['participant_number'])
    del tom_trees_df['player_uuids_str']

    tom_trees_dict = tom_trees_df.to_dict('list')
    choice_rts, prediction_rts = [], []
    move_stay_map = {'A': 'stay', 'B': 'move', 'BA': 'stay', 'BB': 'move'}

    new_col_names = list(analysis_cols.keys())
    n_rows = len(tom_trees_dict['title'])
    for idx in range(n_rows):
        a1 = tom_trees_dict['payoffs_A_p0'][idx]
        b1 = tom_trees_dict['payoffs_BA_p0'][idx]
        c1 = tom_trees_dict['payoffs_BBA_p0'][idx]
        d1 = tom_trees_dict['payoffs_BBB_p0'][idx]
        a2 = tom_trees_dict['payoffs_A_p1'][idx]
        b2 = tom_trees_dict['payoffs_BA_p1'][idx]
        c2 = tom_trees_dict['payoffs_BBA_p1'][idx]
        d2 = tom_trees_dict['payoffs_BBB_p1'][idx]
        payoffs_plr_1 = (a1, b1, c1, d1)
        payoffs_plr_2 = (a2, b2, c2, d2)

        title, game = classify_game(payoffs_plr_1=payoffs_plr_1, payoffs_plr_2=payoffs_plr_2)
        for col_name in new_col_names:
            analysis_cols[col_name].append(game[col_name])

        "choices/predictions"
        choice_p1   = tom_trees_dict['choice__p0'][idx]
        predict_p1  = tom_trees_dict['prediction_B_p0'][idx]
        choice_val  = tom_trees_dict['choice_data__p0'][idx]
        pred_val    = tom_trees_dict['prediction_data_B_p0'][idx]

        "Decode JSON-ish cells"
        if pd.isna(choice_val):
            choice_data = None
        else:
            try:
                choice_data = ast.literal_eval(choice_val)
            except ValueError:
                choice_data = choice_val 

        if pd.isna(pred_val):
            predict_data = None
        else:
            try:
                predict_data = ast.literal_eval(pred_val)
            except ValueError:
                try:
                    predict_data = dict(pred_val)
                except (ValueError, TypeError):
                    predict_data = None

        choice_p1_abdicated = (choice_data is None) or (choice_data.get('rtype') is None if isinstance(choice_data, dict) else True)
        predict_p1_abdicated = (predict_data is None) or ((predict_data.get('rtype') is None) and choice_p1 == 'A' if isinstance(predict_data, dict) else True)

        prediction_rt = None if (predict_data is None or predict_data.get('rtype') is None) else predict_data['rtimedn']
        choice_rt     = None if (choice_data is None   or choice_data.get('rtype')   is None) else choice_data['rtimedn']

        choice = 'none' if choice_p1_abdicated else move_stay_map[choice_p1]
        if choice_p1 == 'A':
            predict = 'rand'
        else:
            predict = 'none' if predict_p1_abdicated else move_stay_map[predict_p1]

        response_cols['choice'].append(choice)
        response_cols['prediction'].append(predict)
        prediction_rts.append(prediction_rt)
        choice_rts.append(choice_rt)

    tom_trees_dict['choice_rts'] = choice_rts
    tom_trees_dict['prediction_rts'] = prediction_rts

    tom_trees_df = pd.DataFrame({**analysis_cols, **response_cols, **tom_trees_dict})
    tom_trees_df = tom_trees_df.drop(columns=['choice_data__p0', 'prediction_data_B_p0'])

    "Save the full analyzed CSV first"
    tom_trees_df.to_csv(full_path, encoding='utf-8-sig')

    "--- Exclusion based on abdications at the ROOT ---"
    abd_by_participant = (
        tom_trees_df.assign(_abd=lambda d: d['choice'].astype(str).str.lower().eq('none'))
                    .groupby('participant_number')['_abd']
                    .sum().astype(int)
    )
    if print_:
        dist = abd_by_participant.value_counts().sort_index()
        total_ppl = dist.sum()
        abdication_counts = ""
        print("\nN Abdications Per Participant Distribution (Root Node Choices):")
        for n_abd, n_ppl in dist.items():
            pct = 100.0 * n_ppl / total_ppl
            abdication_counts += f"({n_abd}) {n_ppl} [{pct:.1f}%]  "
        print(abdication_counts)

    keepers = abd_by_participant[abd_by_participant <= max_abdications].index
    excluded = abd_by_participant[abd_by_participant > max_abdications]
    n_keepers, n_excluded = len(keepers), len(excluded)
    if print_ and n_excluded:
        print("\nExcluded participants for excessive abdications (> {}):".format(max_abdications))
        for pn, n_abd in excluded.items():
            print(f"  participant {pn}: abdications = {n_abd}")

    tom_trees_df = tom_trees_df[tom_trees_df['participant_number'].isin(keepers)].copy()
    tom_trees_df.reset_index(drop=True, inplace=True)

    if print_:
        print("\nFiltered analyzed dataframe (after exclusion):")
        print(tom_trees_df.head(6))
        print("...  (rows:", len(tom_trees_df), ")")

    primary_cols_, analysis_cols_, response_cols_ = {}, {}, {}
    for key in tom_trees_df.columns:
        if key in response_cols.keys():
            response_cols_[key] = tom_trees_df[key].tolist()
        elif key in analysis_cols.keys():
            analysis_cols_[key] = tom_trees_df[key].tolist()
        else:
            primary_cols_[key] = tom_trees_df[key].tolist()

    return {
        'primary_cols': primary_cols_,
        'analysis_cols': analysis_cols_,
        'response_cols': response_cols_,
        'df': tom_trees_df,
    }


def binary_bootstrap_confidence_intervals(data: dict) -> dict:
    """
    Calculates 95% confidence intervals from binary response data.

    Arguments:
        • data: dict; {
            'mistakes': int; The number of mistaken responses,
            'correct': int: The number of correct responses
        }

    Returns:
        • dict {
            'bounds': (float, float): The lower and upper boundaries of the 95% CI,
            'mean': The mean of the bootstrapped error rates,
            'std': The standard deviation of the error rates,
        }
    """
    "Create an array representing all responses: 1 for mistake, 0 for correct, and exclude abdicated"
    responses = np.array([1] * data['mistaken'] + [0] * data['correct'])

    "Number of bootstrap samples"
    n_bootstrap_samples = 10000
    bootstrap_error_rates = []

    "Perform bootstrapping"
    for _ in range(n_bootstrap_samples):
        "Sample with replacement from the responses"
        bootstrap_sample = np.random.choice(responses, size=len(responses), replace=True)
        "Calculate the error rate for this bootstrap sample"
        bootstrap_error_rate = np.mean(bootstrap_sample)
        "Store the error rate"
        bootstrap_error_rates.append(bootstrap_error_rate)

    "Calculate the 95% confidence interval from the bootstrapped error rates"
    lower_bound = np.percentile(bootstrap_error_rates, 2.5)
    upper_bound = np.percentile(bootstrap_error_rates, 97.5)

    return {
        'bounds': (lower_bound, upper_bound),
        'mean': np.mean(bootstrap_error_rates),
        'std': np.std(bootstrap_error_rates),
    }


def count_mistakes(analyzed_data: dict = cleaned_df(), by_participant: bool = False, confidence_intervals: bool = True, print_: bool = False) -> dict:
    """
    Counts data points for the four game categories (and quadruplets), such as error rates and abdicated.

    Four Game Categories:
    1. ⊥⊥ - (SMSM, MSMS); BM labels as diagnostic, and NM labels as diagnostic: Both models predict a high error rates.
    2. ⊥⊤ - (SMSS, MSMM); BM labels as diagnostic, and NM labels as nondiagnostic: Only the BM predicts a high error rates.
    3. ⊤⊥ - (SSSM, MMMS); BM labels as nondiagnostic, and NM labels as diagnostic: Only the BM predicts a high error rates.
    4. ⊤⊤ - (SSSS, MMMM); BM labels as nondiagnostic, and NM labels as nondiagnostic: Neither model predicts high error rates.

    Arguments:
        • analyzed_data: dict; Preprocessed data.
        • by_participant: bool; If true, reports individual differences.
        • confidence_intervals: bool; If true, includes 95% CIs.
        • print_: bool; If true, prints to terminal.

    Returns:
        • dict: Counts of data points subdivided into simple error rates by game type and more detailed data by quadruplet. 
    """
    def count_mistakes_(analyzed_data: dict = cleaned_df(), confidence_intervals: bool = True, print_: bool = True) -> dict:
        """
        Helper function for count_mistakes().  
        """
        mistakes = {'⊥⊥': 0, '⊥⊤': 0, '⊤⊥': 0, '⊤⊤': 0}
        mistakes_ = {'⊥⊥': 0, '⊥⊤': 0, '⊤⊥': 0, '⊤⊤': 0}
        quadruplets = {
            'SMSM': {'etype': '⊥⊥', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'MSMS': {'etype': '⊥⊥', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'SMSS': {'etype': '⊥⊤', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'MSMM': {'etype': '⊥⊤', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'SSSM': {'etype': '⊤⊥', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'MMMS': {'etype': '⊤⊥', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'SSSS': {'etype': '⊤⊤', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0},
            'MMMM': {'etype': '⊤⊤', 'move': 0, 'stay': 0, 'none': 0, 'correct': 0, 'mistaken': 0, 'abdicated': 0, 'total': 0, 'error_rate': 0}
        }

        analysis_cols = analyzed_data['analysis_cols']
        response_cols = analyzed_data['response_cols']
        primary_cols = analyzed_data['primary_cols']
        tom_trees_df = analyzed_data['df']

        df_dict = tom_trees_df.to_dict('list')
        n_rows = n_rows = len(response_cols['choice'])

        for idx in range(n_rows):
            quadruplet = df_dict['quadruplet'][idx]
            mistake_type = df_dict['mistake'][idx]
            choice = df_dict['choice'][idx]
            quadruplets[quadruplet][choice] += 1
            quadruplets[quadruplet]['total'] += 1

            if choice in ['move', 'stay']:
                if choice == df_dict['BM2'][idx]:
                    "correct choice"
                    quadruplets[quadruplet]['correct'] += 1
                else:
                    "mistaken choice"
                    quadruplets[quadruplet]['mistaken'] += 1
                    mistakes[mistake_type] += 1
            else:
                "abdicated choice"
                quadruplets[quadruplet]['abdicated'] += 1

        for quadruplet in list(quadruplets.keys()):
            try:
                quadruplets[quadruplet]['error_rate'] = round(quadruplets[quadruplet]['mistaken'] / \
                    (quadruplets[quadruplet]['correct'] + quadruplets[quadruplet]['mistaken']), 5)
            except ZeroDivisionError:
                quadruplets[quadruplet]['error_rate'] = None

            if confidence_intervals:
                "Confidence Intervals Arround Error Rate"
                ci_dict = binary_bootstrap_confidence_intervals(quadruplets[quadruplet])
                bounds = (round(ci_dict['bounds'][0], 5), round(ci_dict['bounds'][1], 5))
                quadruplets[quadruplet]['error_ci_low'] = bounds[0]
                quadruplets[quadruplet]['error_ci_high'] = bounds[1]

            if print_: 
                print(quadruplet, quadruplets[quadruplet])

            mistake_type = quadruplets[quadruplet]['etype']

            if not isinstance(mistakes_[mistake_type], dict):
                mistakes_[mistake_type] = copy.deepcopy(quadruplets[quadruplet])
            else:
                for key, val in list(quadruplets[quadruplet].items()):
                    if val:
                        mistakes_[mistake_type][key] += val

                if isinstance(mistakes_[mistake_type]['error_rate'], (int, float)):
                    mistakes_[mistake_type]['error_rate'] = round(mistakes_[mistake_type]['error_rate']/2, 5)
                mistakes_[mistake_type]['etype'] = mistakes_[mistake_type]['etype'][:2]

        if confidence_intervals:
            "Confidence Intervals Arround Error Rate"
            for mist in mistakes_.values():
                ci_dict = binary_bootstrap_confidence_intervals(mist)
                bounds = (round(ci_dict['bounds'][0], 5), round(ci_dict['bounds'][1], 5))
                mist['error_ci_low'], mist['error_ci_high'] = bounds[0], bounds[1]

        if print_:
            print("")
            for mistake_type in list(mistakes_.keys()):
                print(mistake_type, mistakes_[mistake_type])

            print(""), print(mistakes), print("")

        return {**mistakes_, **quadruplets}

    tom_trees_df = analyzed_data['df']

    if not by_participant:
        return count_mistakes_(analyzed_data, confidence_intervals, print_)
    
    results = {}
    n_participants = len(list(set(tom_trees_df['participant_number'])))
    for idx in range(1, n_participants + 1):
        player_key = f"player_{idx}"
        if print_:
            print(player_key)
            
        df = tom_trees_df.loc[tom_trees_df['participant_number'] == idx]

        results[player_key] = count_mistakes_(df, print_)

    return results


def error_results_df(results: dict = count_mistakes(cleaned_df()), recreate_csv: bool = True) -> pd.DataFrame:
    """
    Returns error rates by game group and quadruplet across all participants.
    """
    file_name = "Morality_Game_Study_Results_Higher_ToM_ERates.csv"
    full_path = os.path.join(file_path_clean, file_name)

    def _ensure_game_type_col(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'game_type' in df.columns and df.index.name == 'game_type':
            df.index.name = None
            return df.reset_index(drop=True)

        if 'game_type' in df.columns:
            return df.reset_index(drop=True)

        if df.index.name == 'game_type':
            df = df.reset_index()  # brings index into a 'game_type' column
            return df

        if 'Unnamed: 0' in df.columns:
            if df['Unnamed: 0'].astype(str).isin(['⊥⊥','⊥⊤','⊤⊥','⊤⊤']).all():
                df = df.rename(columns={'Unnamed: 0': 'game_type'})
                return df.reset_index(drop=True)

        return df.reset_index(drop=True)

    if recreate_csv or not os.path.exists(full_path):
        results_df = pd.DataFrame(results).transpose()
        results_df.index.name = 'game_type'
        results_df = _ensure_game_type_col(results_df)
        results_df.to_csv(full_path, encoding='utf-8-sig', index=False)
        return results_df

    results_df = pd.read_csv(full_path)
    if 'Unnamed: 0' in results_df.columns:
        del results_df['Unnamed: 0']
    results_df = _ensure_game_type_col(results_df)

    if 'game_type' not in results_df.columns:
        raise KeyError("Invariant violated: 'game_type' not found as a column after normalization.")
    if results_df.index.name is not None:
        results_df.index.name = None

    return results_df


"""
===================================================================================
=========================== SIMULATING ALTERNATIVE MODEL ==========================
===================================================================================
"""
def softmax_(uA: float, uB: float, temperature: float = 1.0, max_temperature: float | None = 1e6, 
             use_fallback: bool = True, preferences_are_ordinal: bool = False) -> float:
    """
    Convert utilities for A and B into a probability of choosing A using softmax.
    
    Arguments:
        • uA: float; utility for choosing A over B
        • uB: float; utility for choosing B over A (or just -uA if symmetrical)
        • temperature: float; stochasticity of the choice probability

    Returns:
        • float; probability of choosing A
    """
    "Ensure temperature is valid"
    if temperature < 0:
        raise ValueError(f"Temperature must be greater than or equal to 0. Received: {temperature}.")
    
    "Check if utilities are finite"
    uA_finite = np.isfinite(uA)
    uB_finite = np.isfinite(uB)

    if not uA_finite and not uB_finite:
        "Both utilities are non-finite → No meaningful comparison, return 0.5."
        if use_fallback:
            print(f"Warning: Both uA and uB are non-finite (uA={uA}, uB={uB}). Returning 0.5.")
            return 0.5
        else:
            raise ValueError(f"Both utilities are non-finite: uA={uA}, uB={uB}")

    elif uA_finite and not uB_finite:
        "Only uA is finite → Compare to zero."
        if use_fallback:
            print(f"Warning: uB is non-finite (uB={uB}). Using uA={uA} for decision.")
            return 1.0 if uA > 0 else 0.0
        else:
            raise ValueError(f"Non-finite uB: {uB}")

    elif not uA_finite and uB_finite:
        "Only uB is finite → Compare to zero."
        if use_fallback:
            print(f"Warning: uA is non-finite (uA={uA}). Using uB={uB} for decision.")
            return 0.0 if uB > 0 else 1.0
        else:
            raise ValueError(f"Non-finite uA: {uA}")

    "Choice probabilities become completely random after the max temp is reached."
    if isinstance(max_temperature, (float, int)) and \
        (max_temperature > 0.0) and (temperature >= max_temperature):
        return 0.5

    "Choice probabilies are certain when the temperature equals zero."
    if temperature == 0:
        if uA > uB: return 1.0
        elif uA < uB: return 0.0
        else: return 0.5

    "If preferences are ordinal, make abs(uA - uB) == 1."
    if preferences_are_ordinal:
        if uA > uB:
            uA, uB = 1, 0
        elif uA < uB:
            uA, uB = 0, 1

    "Create a NumPy array of the utilities, scaled by temperature"
    utilities = np.array([uA, uB]) / temperature

    "Use scipy.special.softmax to calculate the probabilities"
    probabilities = softmax(utilities)

    "Return the probability of choosing A (the first element)"
    return probabilities[0]


def random_primary_models(payoffs_plr_1: tuple[int], payoffs_plr_2: tuple[int], p_lapse_to_k1: float, model: str) -> str:
    """
    Simulates the choices of the NM, BMc, BMd, and RAM with a probability of reverting to lower-order reasoning.

    Arguments:
        • payoffs_plr_1: tuple[int]; Player 1's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_plr_2: tuple[int]; Player 2's ordinal payoffs in the 3-step Stackelberg game
        • p_lapse_to_k1: float ∈ [0, 1]; Probability that the participant incorrectly represents 
            the payoffs in nodes C and D, resulting in some payoffs being transposed.
        • ignore_c_heuristic: bool; Normally, the BM ignores node D, but if this 
            argument is true, then it will ablate node C instead of node D.

    Returns:
        • choice: 'move' | 'stay'; Player 1's optimal choice at the root node.    
    """
    if not (0.0 <= float(p_lapse_to_k1) <= 1.0):
        raise ValueError(f"p_lapse_to_k1 must be in [0, 1], not {p_lapse_to_k1}.")

    ToM_level = 1 if random.random() < p_lapse_to_k1 else 2

    if model == 'NM':
        return stay_or_move(payoffs_plr_1=payoffs_plr_1, payoffs_plr_2=payoffs_plr_2, model_type='NM', ToM_level=ToM_level)

    if model == 'BMd':
        ignore_c_heuristic = False
    elif model == 'BMc':
        ignore_c_heuristic = True
    else:
        ignore_c_heuristic = random.choice([True, False])

    if ignore_c_heuristic:
        "If true, the model will ablate node C instead of node D."       
        payoffs_plr_1 = (payoffs_plr_1[0], payoffs_plr_1[1], payoffs_plr_1[3], payoffs_plr_1[2])
        payoffs_plr_2 = (payoffs_plr_2[0], payoffs_plr_2[1], payoffs_plr_2[3], payoffs_plr_2[2])

    return stay_or_move(payoffs_plr_1=payoffs_plr_1, payoffs_plr_2=payoffs_plr_2, model_type='BM', ToM_level=ToM_level)


def payoff_corruption_model(payoffs_plr_1: tuple[int], payoffs_plr_2: tuple[int], payoffs_to_switch: tuple[bool | int] = (0, 0, 0, 0, 0, 0, 0, 0), 
                            payoffs_to_random: tuple[bool | int] = (0, 0, 0, 0, 0, 0, 0, 0), default_payoffs: list[int] = [1, 2, 3, 4], p_lapse_to_k1: float = 1.0) -> str:
    """
    Model of ToM+ where lower-order representations are corrupted via transposing or randomizing payoffs.

    Arguments:
        • payoffs_plr_1: tuple[int]; Player 1's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_plr_2: tuple[int]; Player 2's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_to_random: tuple[bool]; Payoffs to randomize: a1, b1, c1, d1, a2, b2, c2, d2
        • payoffs_to_switch: tuple[bool]; Payoffs to transpose: a1, b1, c1, d1, a2, b2, c2, d2
        • default_payoffs: list[int]; Payoffs to randomly select from. Determines the mean and 
            variance of the default values that replace the corrupted payoffs.
        • p_lapse_to_k1: float ∈ [0, 1]; Probability of reverting to lower-order reasoning
            
    Returns:
        • choice: 'move' | 'stay'; Player 1's choice at the root node.       
    """
    "List of all payoffs in the order: a1, b1, c1, d1, a2, b2, c2, d2"
    all_payoffs = list(payoffs_plr_1) + list(payoffs_plr_2)

    "Payoffs are only corrupted when higher-order reasoning becomes infeasible."
    if isinstance(p_lapse_to_k1, (int, float)) and (0.0 <= p_lapse_to_k1 <= 1.0):
        if random.random() < p_lapse_to_k1:

            "Finding indices of payoffs to transpose. Must be exactly two."
            payoffs_to_switch = [bool(flip) for flip in payoffs_to_switch]
            flip_indices = [idx for idx, flip in enumerate(payoffs_to_switch) if flip]
            if flip_indices:
                if len(flip_indices) != 2:
                    raise ValueError(f"Exactly two payoffs should flip!")
                
                "Transposing payoffs"
                all_payoffs[flip_indices[0]], all_payoffs[flip_indices[1]] = \
                    all_payoffs[flip_indices[1]], all_payoffs[flip_indices[0]]
            
            "Randomizing payoffs"
            payoffs_to_random = [bool(rand) for rand in payoffs_to_random]
            for idx in range(min(len(all_payoffs), len(payoffs_to_random))):
                if payoffs_to_random[idx]:
                    all_payoffs[idx] = random.choice(default_payoffs)

    "Assigning final payoff values."
    a1, b1, c1, d1, a2, b2, c2, d2 = all_payoffs

    "Solve based on optimal higher-order reasoning."
    pay_plr_1_cd = max(c1, d1)
    pay_plr_2_cd = c2 if c1 > d1 else d2
    pay_plr_1_bcd = b1 if b2 > pay_plr_2_cd else pay_plr_1_cd

    "Return the final choice"
    return 'move' if pay_plr_1_bcd > a1 else 'stay'


def noise_propagation_model(payoffs_plr_1: tuple[int, int, int, int], payoffs_plr_2: tuple[int, int, int, int], 
                            p_lapse_to_k1: float, choice_noise_root: float = 1.2, collapse_to_binary_choice: bool = False, 
                            preferences_are_ordinal: bool = False, depth_schedule: str = "exp", growth: float = 2.0, 
                            k2_noise_scale: float = 0.5) -> str:
    """
    Depth-graded fuzziness model (noise propagation).

    At each choice point in the subtree, the agent compares options with a SoftMax
    whose temperature grows with depth (“fog with distance”). When k=2 (no lapse),
    temperatures are scaled down by `k2_noise_scale`; when k=1 they are larger.

    Resolution policy:
        • If collapse_to_binary_choice = False (default): propagate *expected* values
            upward: use the SoftMax probability to form E[C/D] for each player, then
            compare B vs E[C/D] (SoftMax), then A vs E[B, C/D] (SoftMax).
        • If collapse_to_binary_choice = True: at each node, *sample* a single branch
            according to the SoftMax probability and discard the other (“pruning”).

    Arguments:
        • payoffs_plr_1, payoffs_plr_2 : (A1,B1,C1,D1), (A2,B2,C2,D2)
            Ordinal payoffs (1–4) in the 3-step Stackelberg game.
        • p_lapse_to_k1 : float in [0,1]
            Probability of a lapse to lower-order reasoning (k=1).
        • choice_noise_root : float >= 0
            Base SoftMax temperature at the root (depth 0). Deeper nodes scale up.
        • collapse_to_binary_choice : bool
            False = blur by averaging (expected values). True = blind sampling.
        • depth_schedule : {"exp","lin"}
            "exp": temp(depth) = choice_noise_root * (growth ** depth)
            "lin": temp(depth) = choice_noise_root * (1.0 + growth * depth)
        • growth : float > 0
            Depth growth parameter (see schedule).
        • k2_noise_scale : float in (0,1]
            Factor applied to all temperatures when k=2 (no lapse).

    Returns:
        • 'move' or 'stay' : str
            Root choice for Player 1.
    """
    "Validate inputs"
    if not (0.0 <= float(p_lapse_to_k1) <= 1.0):
        raise ValueError(f"p_lapse_to_k1 must be in [0,1], got {p_lapse_to_k1}")
    if choice_noise_root < 0:
        raise ValueError(f"choice_noise_root must be >=0, got {choice_noise_root}")
    if depth_schedule not in {"exp", "lin"}:
        raise ValueError(f"depth_schedule must be 'exp' or 'lin', got {depth_schedule}")
    if growth <= 0:
        raise ValueError(f"growth must be >0, got {growth}")
    if not (0.0 < k2_noise_scale <= 1.0):
        raise ValueError(f"k2_noise_scale must be in (0,1], got {k2_noise_scale}")

    a1, b1, c1, d1 = payoffs_plr_1
    a2, b2, c2, d2 = payoffs_plr_2

    "lapse: k=1 → larger temps; k=2 → scaled temps"
    k = 1 if (random.random() < p_lapse_to_k1) else 2
    k_scale = 1.0 if k == 1 else k2_noise_scale

    def temp_at_depth(depth: int) -> float:
        base = choice_noise_root
        if depth_schedule == "exp":
            t = base * (growth ** depth)
        else:  # "lin"
            t = base * (1.0 + growth * depth)
        return t * k_scale

    "Bottom: (P1 chooses C vs D)"
    tau_cd = temp_at_depth(2)
    p_choose_c = softmax_(uA=c1, uB=d1, temperature=tau_cd, 
                          preferences_are_ordinal=preferences_are_ordinal)

    if collapse_to_binary_choice:
        choose_c = (random.random() < p_choose_c)
        p1_cd = c1 if choose_c else d1
        p2_cd = c2 if choose_c else d2
    else:
        p1_cd = p_choose_c * c1 + (1 - p_choose_c) * d1
        p2_cd = p_choose_c * c2 + (1 - p_choose_c) * d2

    "Middle: (P2 chooses B vs C/D)"
    tau_b = temp_at_depth(1)
    p_choose_b = softmax_(uA=b2, uB=p2_cd, temperature=tau_b, 
                          preferences_are_ordinal=preferences_are_ordinal)
    
    if collapse_to_binary_choice:
        choose_b = (random.random() < p_choose_b)
        p1_bcd = b1 if choose_b else p1_cd
    else:
        p1_bcd = p_choose_b * b1 + (1 - p_choose_b) * p1_cd

    "Root: (P1 chooses A vs B/CD)"
    tau_a = temp_at_depth(0)
    p_choose_a = softmax_(uA=a1, uB=p1_bcd, temperature=tau_a, 
                          preferences_are_ordinal=preferences_are_ordinal)
    
    return 'move' if (random.random() > p_choose_a) else 'stay'


def data_fit(g1: float, g2: float, g3: float, g4: float, observed_error_distribution: dict[int: float]) -> float:
    total_errors = sum([g1, g2, g3, g4])
    if total_errors <= 0:
        g1, g2, g3, g4 = 0.0, 0.0, 0.0, 0.0
    else:
        g1, g2 = g1 / total_errors, g2 / total_errors
        g3, g4 = g3 / total_errors, g4 / total_errors
    g1_hat, g2_hat = observed_error_distribution[1], observed_error_distribution[2]
    g3_hat, g4_hat = observed_error_distribution[3], observed_error_distribution[4]
    return 1 - (abs(g1_hat - g1) + abs(g2_hat - g2) + abs(g3_hat - g3) + abs(g4_hat - g4)) 


def _game_meta_from_selected(selected_trees: list[tuple[int, ...]]) -> list[dict]:
    """
    Build a list of dicts containing per-game metadata for simulation:
      - payoffs for P1 and P2
      - correct 2nd-order optimal choice
      - mistake group flag ('⊥⊥','⊥⊤','⊤⊥','⊤⊤') from classify_game()
    """
    games = []
    for tup in selected_trees:
        a1, b1, c1, d1, a2, b2, c2, d2 = tup
        p1 = (a1, b1, c1, d1)
        p2 = (a2, b2, c2, d2)
        "classify_game() returns dict w/ BM2,NM2,mistake, etc."
        _, game = classify_game(p1, p2)  # uses your function
        "Correct ToM2 optimal choice (BM2==NM2):"
        correct = game['BM2']  # same as g['NM2']
        games.append({
            'p1': p1, 'p2': p2, 'correct': correct, 'mistake_flag': game['mistake']
        })
    return games


def simulate_alternative_models(n_iters: int = 200, rand_seed: int | None = 42, p_lapse_to_k1_range: list[float] | None = None, 
                                rand_payoff_models: bool = False, rand_switch_models: bool = False, noise_prop_model: bool = True, 
                                export_csv: bool = True, file_name: str = "Alternative_ToM+_Model_Simulation_Results.csv", print_: bool = True) -> pd.DataFrame:
    """
    Simulates other models to see if they better fit the observed distribution of errors. 

    Arguments:
        • n_iters: int; Number of iterations per setting
        • rand_seed: int; Random seed for the simulation
        • p_lapse_to_k1_range: list[float]; Probabilities of lapsing to k=1 thinking
        • rand_payoff_models: bool; If True, runs the Random Defaults Models (RDM).
        • rand_switch_models: bool; If True, runs the Random Transpose Models (RTM).
        • noise_prop_model: bool; If True, runs the Noise Propigation Model (NPM).
        • export_csv: bool; If true, saves the results in a CSV file. 
        • file_name: str; Base of the CSV file name for simulation results. 
        • print_: bool: If true, prints results to terminal.

    Returns:
        • pd.DataFrame: Dataframe of results.
    """
    def compute_error_proportions(total_errors, gcounts) -> dict[int: float]:
        "Compute proportions of errors in each game group"
        if total_errors > 0:
            error_proportions = {game_group: gcounts[game_group] / total_errors for game_group in [1, 2, 3, 4]}
        else:
            error_proportions = {game_group: 0.0 for game_group in [1, 2, 3, 4]} 
        return error_proportions      

    def append_row(rows, n_iters, model, collapse_to_binary_choice, depth_schedule, 
                    base_noise, prob, total_errors, fit, error_proportions, gcounts) -> None:
        rows.append({
            'iters': n_iters,
            'model': model,
            'collpase': collapse_to_binary_choice,
            'depth_schedule': depth_schedule,
            'base_noise': base_noise,
            '𝑝(𝑘=1)': float(prob),
            'G1': int(gcounts[1]),
            'G2': int(gcounts[2]),
            'G3': int(gcounts[3]),
            'G4': int(gcounts[4]),
            'G1_share': round(error_proportions[1], 6),
            'G2_share': round(error_proportions[2], 6),
            'G3_share': round(error_proportions[3], 6),
            'G4_share': round(error_proportions[4], 6),
            'total_errors': int(total_errors),
            'fit': fit
        })   
        return rows  

    def print_row(model, collapse_to_binary_choice, depth_schedule, base_noise, 
                  prob, total_errors, fit, error_proportions, gcounts) -> None:
        print(f"[model={model}] CH={int(collapse_to_binary_choice)} "
            f"DS={depth_schedule} BN={base_noise:.2f} 𝑝(𝑘=1)={prob:.2f} | "
            f"G1={gcounts[1]:07d} ({error_proportions[1]:.4f})  "
            f"G2={gcounts[2]:07d} ({error_proportions[2]:.4f})  "
            f"G3={gcounts[3]:07d} ({error_proportions[3]:.4f})  "
            f"G4={gcounts[4]:07d} ({error_proportions[4]:.4f})  "
            f"N={total_errors:07d} fit={fit:.6f}")        

    """
    Map mistake flag ('⊥⊥','⊥⊤','⊤⊥','⊤⊤') to group index 1..4.
    G1: ⊥⊥  (both diagnostic)     ->  expect high in both models
    G2: ⊥⊤  (BM-only diagnostic)  ->  BM side
    G3: ⊤⊥  (NM-only diagnostic)  ->  NM side
    G4: ⊤⊤  (neither diagnostic)  ->  low   
    """ 
    mistake_group_to_index = {'⊥⊥': 1, '⊥⊤': 2, '⊤⊥': 3, '⊤⊤': 4}
    participant_error_distribution = {1: 0.309, 2: 0.175, 3: 0.309, 4: 0.215}

    if p_lapse_to_k1_range is None:
        p_lapse_to_k1_range = np.round(np.linspace(0, 1, 11), 4).tolist()

    if rand_seed is not None:
        random.seed(rand_seed)
        np.random.seed(rand_seed)

    "Prepare per-game meta (payoffs, correct choice, group)"
    games = _game_meta_from_selected(selected_trees)  

    if print_:
        print(f"Simulating Error Rate Distribution Between Alternative Models:")

    rows = []
    if noise_prop_model:
        collapse_to_binary_choice = False
        for depth_schedule in ('lin', 'exp'):
            for base_noise in (0.15, 0.30, 0.60, 1.20, 2.40): 
                for prob in p_lapse_to_k1_range:
                    gcounts = {1: 0, 2: 0, 3: 0, 4: 0}
                    total_errors = 0

                    for _ in range(n_iters):
                        for game in games:
                            p1, p2 = game['p1'], game['p2']
                            correct = game['correct']
                            gidx = mistake_group_to_index.get(game['mistake_flag'])
                            if gidx is None:
                                raise ValueError(f"Unknown mistake flag: {game['mistake_flag']}")
                            
                            choice = noise_propagation_model(p1, p2, prob, choice_noise_root=base_noise, 
                                                             collapse_to_binary_choice=collapse_to_binary_choice,
                                                             depth_schedule=depth_schedule, preferences_are_ordinal=False)  

                            if choice != correct:
                                gcounts[gidx] += 1
                                total_errors += 1

                    error_proportions = compute_error_proportions(total_errors=total_errors, gcounts=gcounts)

                    fit = data_fit(g1=gcounts[1], g2=gcounts[2], g3=gcounts[3], g4=gcounts[4], 
                                observed_error_distribution=participant_error_distribution)

                    rows = append_row(rows=rows, n_iters=n_iters, model='NPM', collapse_to_binary_choice=collapse_to_binary_choice, depth_schedule=depth_schedule, 
                                      base_noise=base_noise, prob=prob, total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts)

                    if print_:
                        print_row(model='NPM', collapse_to_binary_choice=collapse_to_binary_choice, depth_schedule=depth_schedule, base_noise=\
                                  base_noise, prob=prob, total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts)   

    elif rand_payoff_models or rand_switch_models:
        if rand_payoff_models:
            models = list(it.product([0, 1], repeat=4))
        else:
            models = sorted(list(set(it.permutations([1, 1, 0, 0]))))
        for model in models:
            gcounts = {1: 0, 2: 0, 3: 0, 4: 0}
            total_errors = 0

            for _ in range(n_iters):
                for game in games:    
                    p1, p2 = game['p1'], game['p2']
                    correct = game['correct']
                    gidx = mistake_group_to_index.get(game['mistake_flag'])
                    if gidx is None:
                        raise ValueError(f"Unknown mistake flag: {game['mistake_flag']}")    
                    
                    if rand_payoff_models:   
                        payoffs_to_random = (0, 0, model[0], model[2], 0, 0, model[1], model[3])
                        choice = payoff_corruption_model(payoffs_plr_1=p1, payoffs_plr_2=p2, payoffs_to_random=payoffs_to_random)            

                    else:
                        payoffs_to_switch = (0, 0, model[0], model[2], 0, 0, model[1], model[3])
                        choice = payoff_corruption_model(payoffs_plr_1=p1, payoffs_plr_2=p2, payoffs_to_switch=payoffs_to_switch)   

                    if choice != correct:
                        gcounts[gidx] += 1
                        total_errors += 1

            error_proportions = compute_error_proportions(total_errors=total_errors, gcounts=gcounts)

            fit = data_fit(g1=gcounts[1], g2=gcounts[2], g3=gcounts[3], g4=gcounts[4], 
                            observed_error_distribution=participant_error_distribution)
            
            rows = append_row(rows=rows, n_iters=n_iters, model=model, collapse_to_binary_choice='x', depth_schedule='x', base_noise='x', 
                              prob=1.0, total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts)

            if print_:
                print_row(model=model, collapse_to_binary_choice=0, depth_schedule=0, base_noise=0, prob=1.0, 
                            total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts)                

    else:    
        for model in ['NM', 'BMc', 'BMd', 'RAM']:
            gcounts = {1: 0, 2: 0, 3: 0, 4: 0}
            total_errors = 0

            for _ in range(n_iters):
                for game in games:
                    p1, p2 = game['p1'], game['p2']
                    correct = game['correct']
                    gidx = mistake_group_to_index.get(game['mistake_flag'])
                    if gidx is None:
                        raise ValueError(f"Unknown mistake flag: {game['mistake_flag']}")

                    choice = random_primary_models(p1, p2, 1.0, model)

                    if choice != correct:
                        gcounts[gidx] += 1
                        total_errors += 1

            error_proportions = compute_error_proportions(total_errors=total_errors, gcounts=gcounts)

            fit = data_fit(g1=gcounts[1], g2=gcounts[2], g3=gcounts[3], g4=gcounts[4], 
                            observed_error_distribution=participant_error_distribution)

            rows = append_row(rows=rows, n_iters=n_iters, model=model, collapse_to_binary_choice='x', depth_schedule='x', base_noise='x', 
                              prob=1.0, total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts)

            if print_:
                print_row(model=model, collapse_to_binary_choice=0, depth_schedule=0, base_noise=0, prob=1.0, 
                            total_errors=total_errors, fit=fit, error_proportions=error_proportions, gcounts=gcounts) 

    df = pd.DataFrame(rows)

    if export_csv:
        if file_name.endswith('.csv'):
            file_name = file_name.replace('.csv', '')
        file_name += f'-{rand_seed}-{n_iters}-{int(rand_payoff_models)}-' 
        file_name += f'{int(rand_switch_models)}-{int(noise_prop_model)}.csv' 
        out_path = os.path.join(file_path_clean, file_name)
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        if print_:
            print(f"Saved simulation results to: {out_path.replace(str(ROOT), '...')}\n")

    return df


def preference_weighting_model(payoffs_plr_1: tuple[int], payoffs_plr_2: tuple[int], p_lapse_to_k1: float, preference_weight_p1: float, 
                               preference_weight_p2: float, choice_temperature: float, collapse_to_binary_choice: bool = True) -> str:
    """
    Simulates the choices of an agent that fails to fully represent preferences when higher-order reasoning becomes infeasible.

    Arguments:
        • payoffs_plr_1: tuple[int]; Player 1's ordinal payoffs in the 3-step Stackelberg game
        • payoffs_plr_2: tuple[int]; Player 2's ordinal payoffs in the 3-step Stackelberg game
        • p_lapse_to_k1: float ∈ [0, 1]; Probability that the participant incorrectly represents 
            the payoffs in nodes C and D, resulting in payoffs being randomized
        • preference_weight_p1: float; Strength of consideration for player 1's preferences 
        • preference_weight_p1: float; Strength of consideration for player 2's preferences 
        • choice_temperature: float; Stochasticity of the choice probability function
        • collapse_to_binary_choice: bool; If true, reduces the parent of nodes C and D to the
            payoffs of either C or D by making a probabilistic binary choice; otherwise weights
            the payoffs based on the choice probabilities and allows p2 to make their best choice.

    Returns:
        • choice: 'move' | 'stay'; Player 1's optimal choice at the root node.    
    """
    if not (0.0 <= float(p_lapse_to_k1) <= 1.0):
        raise ValueError(f"p_lapse_to_k1 must be in [0, 1], not {p_lapse_to_k1}.")

    if not isinstance(preference_weight_p1, (float, int)) or not (-1.0 <= preference_weight_p1 <= 1.0):
        raise ValueError(f"preference_weight_p1 must be a value within [-1, 1], not {preference_weight_p1}.")

    if not isinstance(preference_weight_p2, (float, int)) or not (-1.0 <= preference_weight_p2 <= 1.0):
        raise ValueError(f"preference_weight_p2 must be a value within [-1, 1], not {preference_weight_p2}.")

    if not isinstance(choice_temperature, (float, int)) or not (0.0 <= choice_temperature):
        raise ValueError(f"choice_temperature must be a value within [0, ∞], not {choice_temperature}.")    

    a1, b1, c1, d1 = payoffs_plr_1
    a2, b2, c2, d2 = payoffs_plr_2

    "Lapse to lower-order reasoning if 𝑝(𝑘=1) exceeds a random number."
    if random.random() < p_lapse_to_k1:
        utilityC = preference_weight_p1 * c1 + preference_weight_p2 * c2
        utilityD = preference_weight_p1 * d1 + preference_weight_p2 * d2
        probability_choose_C = softmax_(utilityC, utilityD, choice_temperature)
        # print(f"({c1}, {c2}) ({d1}, {d2}) temp={choice_temperature}; p(c)={probability_choose_C};")
        if collapse_to_binary_choice:
            if random.random() < probability_choose_C:
                pay_plr_1_cd = c1
                pay_plr_2_cd = c2
            else:
                pay_plr_1_cd = d1
                pay_plr_2_cd = d2    

        else:
            pay_plr_1_cd = c1 * probability_choose_C + d1 * (1 - probability_choose_C)
            pay_plr_2_cd = c2 * probability_choose_C + d2 * (1 - probability_choose_C)

        pay_plr_1_bcd = b1 if b2 > pay_plr_2_cd else pay_plr_1_cd
        return 'move' if pay_plr_1_bcd > a1 else 'stay'

    "Solve based on optimal higher-order reasoning."
    pay_plr_1_cd = max(c1, d1)
    pay_plr_2_cd = c2 if c1 > d1 else d2
    pay_plr_1_bcd = b1 if b2 > pay_plr_2_cd else pay_plr_1_cd
    return 'move' if pay_plr_1_bcd > a1 else 'stay'


def simulate_preference_weighting_model(selected_trees: list[tuple[int]], preference_weight_p1_range: list[float] = (-1.0, -0.5, 0.0, 0.5, 1.0), 
                                        preference_weight_p2_range: list[float] = (-1.0, -0.5, 0.0, 0.5, 1.0), choice_temperature_range: list[float] = \
                                            (0.0, 0.5, 1.0, 2.0, 4.0, float('inf')), collapse_to_binary_choice: bool = True, p_lapse_to_k1: float = 1.0, rand_seed: int | None = None, 
                                        n_iters: int = 200, export_csv: bool = True, out_csv: str = "Preference_Weighting_Model_Simulation.csv", print_: bool = True) -> pd.DataFrame:
    """
    Grid-simulate mind-based corruption models and summarize where errors fall.

    For each (w1, w2, τ) triple:
        1) Iterate n_iters times through all 32 games.
        2) At the lapse (probability p(k=1)), resolve the bottom node via either:
           • Binary collapse (collapse_to_binary_choice=True): sample C vs D once,
             then backward induction.
           • Weighted collapse (collapse_to_binary_choice=False): expected-value
             summary with p(C) from SoftMax, P2 compares B2 vs E[C2,D2], then root.
        3) Count errors by game group (G1..G4) and compute error shares per group.
        4) Report NM alignment = (G3 - G2) / (G2 + G3), which is 1.0 when all
           disagreement-game errors live in the NM group (G3) and 0.0 when the two
           disagreement groups carry equal errors.

    Arguments:
        • selected_trees: list[tuple[int]]
            The 32 payoff tuples used in the experiment.
        • preference_weight_p1_range / preference_weight_p2_range: list[float]
            Grid of weights for P1 and P2 in bottom-node utilities.
        • choice_temperature_range: list[float]
            Grid of SoftMax temperatures; include float('inf') to probe the NM
            limiting case for weighted collapse.
        • collapse_to_binary_choice: bool
            Select binary vs weighted collapse semantics.
        • p_lapse_to_k1: float
            Lapse probability. Use 1.0 to isolate the lapse policy; <1.0 just scales
            error counts but does not change error *shares*.
        • seed, n_iters, export_csv, out_csv, print_:
            Standard simulation and I/O settings.

    Returns:
        • pd.DataFrame with columns:
          ['w1','w2','tau','G1','G2','G3','G4','G1_share','G2_share','G3_share',
           'G4_share','total_errors','NMAlign','p_lapse_to_k1','binary_choice']
    """
    MISTAKE_GROUP_TO_INDEX = {'⊥⊥': 1, '⊥⊤': 2, '⊤⊥': 3, '⊤⊤': 4}
    participant_error_distribution = {1: 0.309, 2: 0.175, 3: 0.309, 4: 0.215}

    if isinstance(rand_seed, int):
        np.random.seed(rand_seed)
        random.seed(rand_seed)

    games = _game_meta_from_selected(selected_trees)  

    rows = []
    for choice_temperature in choice_temperature_range:
        for preference_weight_p1 in preference_weight_p1_range:
            for preference_weight_p2 in preference_weight_p2_range:
                "Handle τ = ∞ by using a large number; numerically this gives ~0.5"
                choice_temperature = 1e6 if (choice_temperature == float('inf')) else float(choice_temperature)    

                error_counts_by_game_group = {1: 0, 2: 0, 3: 0, 4: 0}
                total_errors = 0

                for _ in range(n_iters):
                    for game in games:
                        payoffs_plr_1, payoffs_plr_2 = game['p1'], game['p2']
                        correct = game['correct']

                        game_group_idx = MISTAKE_GROUP_TO_INDEX[game['mistake_flag']]
                        choice = preference_weighting_model(payoffs_plr_1=payoffs_plr_1, payoffs_plr_2=payoffs_plr_2, 
                                                            preference_weight_p1=preference_weight_p1, preference_weight_p2=preference_weight_p2, 
                                                            choice_temperature=choice_temperature, collapse_to_binary_choice=collapse_to_binary_choice, 
                                                            p_lapse_to_k1=1.0)
                        if choice != correct:
                            error_counts_by_game_group[game_group_idx] += 1
                            total_errors += 1

                if total_errors > 0:
                    shares = [error_counts_by_game_group[k] / total_errors for k in (1, 2, 3, 4)]
                else:
                    shares = [0.0, 0.0, 0.0, 0.0]

                g1, g2 = error_counts_by_game_group[1], error_counts_by_game_group[2]
                g3, g4 = error_counts_by_game_group[3], error_counts_by_game_group[4]
                fit = data_fit(g1=g1, g2=g2, g3=g3, g4=g4, observed_error_distribution=participant_error_distribution)
                nm_align = data_fit(g1=g1, g2=g2, g3=g3, g4=g4, observed_error_distribution={1: 0.5, 2: 0.0, 3: 0.5, 4: 0.0})

                row = {
                    'w1': preference_weight_p1, 'w2': preference_weight_p2, 'tau': choice_temperature,
                    'G1': error_counts_by_game_group[1], 'G2': error_counts_by_game_group[2], 
                    'G3': error_counts_by_game_group[3], 'G4': error_counts_by_game_group[4], 
                    'G1_share': shares[0], 'G2_share': shares[1], 'G3_share': shares[2], 'G4_share': shares[3],
                    'total_errors': total_errors, 'fit': fit, 'NM_Align': nm_align, 'p_lapse_to_k1': p_lapse_to_k1, 
                    'binary_choice': collapse_to_binary_choice
                }
                rows.append(row)

                if print_:
                    print(
                        f"[w1={preference_weight_p1:+.2f}, w2={preference_weight_p2:+.2f}, τ={choice_temperature:011.3f}] "
                        f"G-shares=({shares[0]:.4f}, {shares[1]:.4f}, {shares[2]:.4f}, {shares[3]:.4f}) "
                        f"Fit={fit:+.4f} NMAlign={nm_align:+.4f} N={total_errors:07d} "
                        f"𝑝(𝑘=1)={p_lapse_to_k1:.2f} binary={collapse_to_binary_choice}"
                    )

    df = pd.DataFrame(rows)
    if export_csv:
        if out_csv.endswith('.csv'):
            out_csv = out_csv.replace('.csv', '')
        out_csv += f'-{len(preference_weight_p1_range)}-{len(preference_weight_p2_range)}-{len(choice_temperature_range)}'
        out_csv += f'-{rand_seed}-{n_iters}-{p_lapse_to_k1:.02f}-{collapse_to_binary_choice}.csv'
        out_path = os.path.join(file_path_clean, out_csv)
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        if print_:
            print(f"Saved grid simulation → {out_path.replace(str(ROOT), '...')}")
    return df


def plot_preference_weight_heatmap(grid_csv_name: str | None = "Preference_Weighting_Model_Simulation.csv", df: pd.DataFrame | None = None, 
                                   metric: str = "fit", tau_value: float | str = float('inf'), fig_lay: dict | None = None, export_fig: bool = True, 
                                   file_name: str = "Preference_Weighting_Heatmap.html", tau_tol: float = 1e-3, top_k: int = 8, 
                                   inf_threshold: float = 1e5, p_lapse_to_k1 = 1.0, rand_seed = None, n_iters = 1000,
                                   preference_weight_p1_range = list(np.linspace(start=-1, stop=1, num=101)), 
                                   preference_weight_p2_range = list(np.linspace(start=-1, stop=1, num=101)), 
                                   choice_temperature_range = (0.0, 1.0,), collapse_to_binary_choice = True) -> go.Figure:
    """
    Render a heatmap over (𝑤₁, 𝑤₂) for a fixed τ slice of the preference-weighting grid.

    Arguments:
        • grid_csv_name or df:
            The CSV produced by simulate_preference_weighting_model, or a preloaded DataFrame.
        • metric:
            "NM_Align" for alignment with the NM error allocation, or "fit" for similarity to the
            empirical four-way error shares.
        • tau_value:
            Fixed temperature slice to visualize. Use float('inf') to select the τ→∞ rows; rows with
            τ >= inf_threshold are treated as ∞.
        • fig_lay, export_fig, file_name:
            Aesthetic/layout dictionary (template, fonts, etc.), and output settings.
        • tau_tol, inf_threshold:
            Matching tolerances for τ selection.
        • top_k:
            How many best cells to print to the terminal for quick inspection.

    Returns:
        • A plotly.graph_objects.Figure handle (also saved to HTML if export_fig=True).
    """
    if df is None:
        if grid_csv_name is None:
            raise ValueError("Provide either df or grid_csv_name.")
        
        if grid_csv_name.endswith('.csv'):
            grid_csv_name = grid_csv_name.replace('.csv', '')
        grid_csv_name += f'-{len(preference_weight_p1_range)}-{len(preference_weight_p2_range)}-{len(choice_temperature_range)}'
        grid_csv_name += f'-{rand_seed}-{n_iters}-{p_lapse_to_k1:.02f}-{collapse_to_binary_choice}.csv'

        full_path = os.path.join(file_path_clean, grid_csv_name)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
        else:
            raise FileNotFoundError(f"File not found: {full_path}")

    "Normalize column names"
    col_map = {col.lower(): col for col in df.columns}
    for col in ["w1", "w2", "tau", metric]:
        if col not in df.columns and col.lower() in col_map:
            df.rename(columns={col_map[col.lower()]: col}, inplace=True)

    "Handle τ selection (treat large τ as ∞)"
    df["tau_is_inf"] = df["tau"] >= inf_threshold
    if tau_value == float('inf') or (isinstance(tau_value, str) and str(tau_value).lower() in {"inf", "infinity"}):
        sub = df[df["tau_is_inf"]].copy()
        tau_label = "∞"
    else:
        sub = df[~df["tau_is_inf"] & (np.abs(df["tau"] - float(tau_value)) <= tau_tol)].copy()
        tau_label = f"{float(tau_value):.0f}"

    if sub.empty:
        available = np.sort(df["tau"].unique())
        raise ValueError(f"No rows match τ={tau_value}. Available τ values: {available}")

    "Pivot to (w1 × w2) grid on the chosen metric"
    w1_vals = np.sort(sub["w1"].unique())
    w2_vals = np.sort(sub["w2"].unique())
    Z = sub.pivot_table(index="w2", columns="w1", values=metric, 
                        aggfunc="mean").reindex(index=w2_vals, columns=w1_vals)
    z_mat = Z.values

    "Build heatmap"
    fig = go.Figure(data=go.Heatmap(
        z=z_mat,
        x=w1_vals,  # x-axis = weight on P2
        y=w2_vals,  # y-axis = weight on P1
        colorbar=dict(
            title=metric.capitalize(),
            tickvals=np.linspace(0, 1, 11),  # 0.0, 0.1, ... 1.0
            ticktext=[f"{val:.1f}" for val in np.linspace(0, 1, 11)]
        ),
        zmin=0.0, zmax=1.0,
        colorscale="Viridis",
        hovertemplate=(
            "𝑤₁:  %{x:.3f}<br>"
            "𝑤₂:  %{y:.3f}<br>"
            "Fit: %{z:.3f}<extra></extra>"
        ),
    ))

    "Titles / layout"
    title_core = f"Fit of Preference Weighting Model by 𝑤₁ & 𝑤₂ (τ={tau_label})"
    fig.update_layout(
        title=title_core, titlefont_size=fig_lay['titlefont_size'] - 16,
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], 
        template=fig_lay['template'], font=fig_lay['font'], 
        showlegend=False, margin=dict(l=605, r=605, t=120, b=100), 
    )
    fig.update_xaxes(title="Preference Weight on Player 1 (𝑤₁)",
                     tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
                     ticktext=['', '-0.5', '0.0', '0.5', '1.0'],
                     scaleanchor='y1', scaleratio=1)
    fig.update_yaxes(title="Preference Weight on Player 2 (𝑤₂)",
                     tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
                     ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0'],
                     scaleanchor='x1', scaleratio=1)

    if export_fig:
        if file_name.endswith('.html'):
            file_name = file_name.replace('.html', '')
        file_name += f'-{len(preference_weight_p1_range)}-{len(preference_weight_p2_range)}-{len(choice_temperature_range)}'
        file_name += f'-{rand_seed}-{n_iters}-{p_lapse_to_k1:.02f}-{collapse_to_binary_choice}-{tau_value}-{metric}.html'
        fig.write_html(os.path.join(file_path_figures, file_name))
        print(f"Saved heatmap → {file_name}")

    "Print top-k cells (by metric)"
    flat = sub[["w1", "w2", metric]].dropna().sort_values(metric, ascending=False)
    print(f"\nTop {min(top_k, len(flat))} cells for {metric}, τ={tau_label}:")
    for idx, row in enumerate(flat.head(top_k).itertuples(index=False), start=1):
        print(f"  {idx:2d}) 𝑤1={row.w1:+.2f}, 𝑤2={row.w2:+.2f}, {metric}={getattr(row, metric):+.4f}")

    return fig


"""
===================================================================================
============================== INDIVIDUAL DIFFERENCES =============================
===================================================================================
"""
def _agg_one_group(df: pd.DataFrame) -> pd.Series:
    """
    Helper for groupby aggregation on a participant x group slice.
    Computes counts, error rate (excluding abdications), and medians of RTs.
    """
    "Non-abdicated root decisions"
    nonabd = df[df['choice'].isin(['move', 'stay'])]

    correct   = int((nonabd['choice'] == nonabd['BM2']).sum())
    mistaken  = int((nonabd['choice'] != nonabd['BM2']).sum())
    abdicated = int((df['choice'] == 'none').sum())
    total     = int(len(df))

    denom = correct + mistaken
    error_rate = float(mistaken / denom) if denom > 0 else float('nan')

    choice_rt_med = pd.to_numeric(nonabd['choice_rts'], errors='coerce').dropna().median()
    pred_rt_med   = pd.to_numeric(df['prediction_rts'], errors='coerce').dropna().median()

    return pd.Series({
        'n_correct': correct,
        'n_mistaken': mistaken,
        'n_abdicated': abdicated,
        'n_total': total,
        'error_rate': error_rate,
        'median_choice_rt': choice_rt_med,
        'median_prediction_rt': pred_rt_med
    })


def make_erates_individual(analyzed_df: pd.DataFrame,
    out_name_groups: str = "Morality_Game_Study_Results_Higher_ToM_ERates_Individual.csv",
    out_name_quads:  str = "Morality_Game_Study_Results_Higher_ToM_ERates_Individual_Quadruplets.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build participant-level error/RT datasets:
      (A) by design group (⊥⊥, ⊥⊤, ⊤⊥, ⊤⊤)
      (B) by quadruplet (SMSM, MSMS, …)

    Returns (df_groups, df_quads) and saves both CSVs in file_path_clean.
    """
    group_code_to_labels = {
        '⊥⊥': {"gid": "G1", "label": "G1: both diagnostic",    "bm": "High", "nm": "High"},
        '⊥⊤': {"gid": "G2", "label": "G2: BM-only diagnostic", "bm": "High", "nm": "Low"},
        '⊤⊥': {"gid": "G3", "label": "G3: NM-only diagnostic", "bm": "Low",  "nm": "High"},
        '⊤⊤': {"gid": "G4", "label": "G4: neither diagnostic", "bm": "Low",  "nm": "Low"},
    }

    df = analyzed_df.copy()

    "(A) Per-participant x group (⊥⊥, ⊥⊤, ⊤⊥, ⊤⊤)"
    df['game_group'] = df['mistake']  # already BM-first then NM
    group = df.groupby(['participant_number', 'game_group'], sort=False).apply(_agg_one_group).reset_index()

    "Labels for display / analysis convenience"
    group['group_id']    = group['game_group'].map(lambda code: group_code_to_labels[code]['gid'])
    group['group_label'] = group['game_group'].map(lambda code: group_code_to_labels[code]['label'])
    group['bm_pred']     = group['game_group'].str[0].map({'⊥': 'High', '⊤': 'Low'})
    group['nm_pred']     = group['game_group'].str[1].map({'⊥': 'High', '⊤': 'Low'})

    "Consistent order for plotting"
    order = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4}
    group['order'] = group['group_id'].map(order)
    group = group.sort_values(['participant_number', 'order']).drop(columns=['order'])

    "Save error rates CSV."
    out_groups = os.path.join(file_path_clean, out_name_groups)
    group.to_csv(out_groups, index=False, encoding='utf-8-sig')
    print(f"Saved participant-level ERates (groups) → {str(out_groups).replace(str(ROOT), '...')}")

    "(B) Per-participant x quadruplet (SMSM, MSMS, …)"
    quads = df.groupby(['participant_number', 'quadruplet'], 
                       sort=False).apply(_agg_one_group).reset_index()
    out_quads = os.path.join(file_path_clean, out_name_quads)
    quads.to_csv(out_quads, index=False, encoding='utf-8-sig')
    print(f"Saved participant-level ERates (quadruplets) → {str(out_quads).replace(str(ROOT), '...')}")

    return group, quads


def add_median_rts_to_erates(analyzed_df: pd.DataFrame, eresults_df: pd.DataFrame,
    out_name: str = "Morality_Game_Study_Results_Higher_ToM_ERates.csv") -> pd.DataFrame:
    """
    Add median choice/prediction RT columns to the aggregated ERates CSV (both groups and quadruplets).

    Writes back to the same CSV name by default.
    """
    "Valid (non-abdicated) choices for choice RTs"
    valid = analyzed_df[analyzed_df['choice'].isin(['move', 'stay'])].copy()

    "Medians by group (use 'mistake' column → 'game_type')"
    med_g_choice = valid.groupby('mistake')['choice_rts'].median().reset_index()
    med_g_choice = med_g_choice.rename(columns={'mistake': 'game_type', 'choice_rts': 'median_choice_rt'})

    med_g_pred = analyzed_df.groupby('mistake')['prediction_rts'].apply(
        lambda s: pd.to_numeric(s, errors='coerce').dropna().median()
    ).reset_index().rename(columns={'mistake': 'game_type', 'prediction_rts': 'median_prediction_rt'})

    med_groups = pd.merge(med_g_choice, med_g_pred, on='game_type', how='outer')

    "Medians by quadruplet (→ 'game_type')"
    med_q_choice = valid.groupby('quadruplet')['choice_rts'].median().reset_index()
    med_q_choice = med_q_choice.rename(columns={'quadruplet': 'game_type', 'choice_rts': 'median_choice_rt'})

    med_q_pred = analyzed_df.groupby('quadruplet')['prediction_rts'].apply(
        lambda s: pd.to_numeric(s, errors='coerce').dropna().median()
    ).reset_index().rename(columns={'quadruplet': 'game_type', 'prediction_rts': 'median_prediction_rt'})

    med_quads = pd.merge(med_q_choice, med_q_pred, on='game_type', how='outer')

    med_all = pd.concat([med_groups, med_quads], ignore_index=True)

    "Merge into the aggregated ERates dataframe"
    updated = pd.merge(eresults_df, med_all, on='game_type', how='left')

    "Save updated CSV"
    out_csv = os.path.join(file_path_clean, out_name)
    updated.to_csv(out_csv, encoding='utf-8-sig', index=False)
    print(f"Appended median RT columns → {out_csv.replace(str(ROOT), '...')}")

    return updated


"""
===================================================================================
============================== REACTION TIME ANALYSIS =============================
===================================================================================
"""
def participant_rt_by_outcome(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per participant, compute median root-node choice RT for 'correct' 
    vs 'mistake'. Returns one row per (participant, outcome).
    """
    df = analyzed_df.copy()
    df = df[df['choice'].isin(['move', 'stay'])].copy()
    df['is_correct'] = (df['choice'] == df['BM2'])
    df['outcome'] = np.where(df['is_correct'], 'Correct', 'Error')
    df['choice_rts'] = pd.to_numeric(df['choice_rts'], errors='coerce')

    group = (df.groupby(['participant_number', 'outcome'], as_index=False)
        .agg(median_choice_rt=('choice_rts', 'median'), n_trials=('choice_rts', 'size')))
    return group


def participant_rt_by_group(analyzed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per participant × game group (G1..G4), compute median root-node choice RT (excluding abdications).
    """
    label_map = {
        '⊥⊥': ('G1', 'G1: both diagnostic'),
        '⊥⊤': ('G2', 'G2: BM-only diagnostic'),
        '⊤⊥': ('G3', 'G3: NM-only diagnostic'),
        '⊤⊤': ('G4', 'G4: neither diagnostic')
    }
    df = analyzed_df.copy()
    df = df[df['choice'].isin(['move', 'stay'])].copy()
    df['group_id']    = df['mistake'].map(lambda s: label_map[s][0])
    df['group_label'] = df['mistake'].map(lambda s: label_map[s][1])
    df['choice_rts']  = pd.to_numeric(df['choice_rts'], errors='coerce')

    group = (df.groupby(['participant_number', 'group_id', 'group_label'], as_index=False)
        .agg(median_choice_rt=('choice_rts', 'median'), n_trials=('choice_rts', 'size')))
    
    "Maintain plotting order"
    order = {'G1': 1, 'G2': 2, 'G3': 3, 'G4': 4}
    group['plot_order'] = group['group_id'].map(order)
    return group.sort_values(['participant_number', 'plot_order']).drop(columns=['plot_order'])


def rt_statistics(rt_outcome_df: pd.DataFrame, rt_group_df: pd.DataFrame) -> None:
    """
    Prints statistics:
        • Wilcoxon signed-rank: participant median RT Error vs Correct.
        • Friedman (and pairwise Wilcoxon with Bonferroni): participant median RTs across G1..G4.
    """
    print("Reaction Time Statistics:")

    "Error vs Correct (paired on participant)"
    wide_oc = (rt_outcome_df
               .pivot(index='participant_number', columns='outcome', values='median_choice_rt')
               .dropna(subset=['Correct', 'Error']))
    
    if len(wide_oc) >= 5:  
        "Avoid tiny-N warnings"
        stat, p = wilcoxon(wide_oc['Error'], wide_oc['Correct'])
        print(f"Wilcoxon (Error >? Correct): W = {stat:.3f}, p = {p:.6f}, N = {len(wide_oc)}")
        print(f"Median(Error) = {np.nanmedian(wide_oc['Error']):.3f}s; "
              f"Median(Correct) = {np.nanmedian(wide_oc['Correct']):.3f}s")
    else:
        print("Not enough paired data for Wilcoxon Error vs Correct.")

    "G1..G4 repeated-measures"
    wide_g = (rt_group_df
              .pivot(index='participant_number', columns='group_id', values='median_choice_rt')
              .reindex(columns=['G1', 'G2', 'G3', 'G4']).dropna())
    
    if len(wide_g) >= 5:
        stat, p = friedmanchisquare(wide_g['G1'], wide_g['G2'], wide_g['G3'], wide_g['G4'])
        print(f"Friedman across groups: χ² = {stat:.3f}, p = {p:.6f}, N = {len(wide_g)}")

        "Pairwise Wilcoxon with Bonferroni"
        pairs = [('G1','G2'), ('G1','G3'), ('G1','G4'), ('G2','G3'), ('G2','G4'), ('G3','G4')]
        pairs_len = len(pairs)
        rows = []
        for group_a, group_b in pairs:
            ok = wide_g[[group_a, group_b]].dropna()
            if len(ok) >= 5:
                wilx, p_val = wilcoxon(ok[group_a], ok[group_b])
                rows.append({'Pair': f'{group_a} vs {group_b}', 'W': wilx, 'p_raw': p_val, 'p_bonf': min(p_val*pairs_len, 1.0),
                             'median_diff': np.nanmedian(ok[group_a]-ok[group_b])})
        if rows:
            print(pd.DataFrame(rows))
            print("")
    else:
        print("Not enough complete G1–G4 data for Friedman.")
        print("")


"""
===================================================================================
====================== Choice-Prediction Consistency Analysis =====================
===================================================================================
"""
def prediction_consistency_by_quadruplet(analyzed_df: pd.DataFrame, round_to: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prints a tidy table of CC/N, CM/N, MC/N, MM/N and N (rows = quadruplets),
    plus an interpretive table with semantic labels. Returns (ratios_df, counts_df, interpret_df).
    """
    def _p2_optimal_choice(row: pd.Series) -> str:
        """
        Player 2's optimal move via backward induction at the middle node.
        Ties default to 'move' (consistent with your stay_or_move implementation).
        """
        c1 = row['payoffs_BBA_p0']; d1 = row['payoffs_BBB_p0']
        b2 = row['payoffs_BA_p1'];  c2 = row['payoffs_BBA_p1']; d2 = row['payoffs_BBB_p1']
        p2_future = d2 if d1 > c1 else c2
        return 'stay' if b2 > p2_future else 'move'

    "Keep only trials where a prediction is observable (root = Move; prediction recorded)"
    df = analyzed_df.copy()
    df = df[(df['choice'] == 'move') & (df['prediction'].isin(['stay','move']))].copy()

    "P2 optimal, prediction correctness, and root correctness"
    df['p2_optimal']   = df.apply(_p2_optimal_choice, axis=1)
    df['pred_correct'] = (df['prediction'] == df['p2_optimal']).astype(int)
    "Given I filtered to 'move', this is 1 (M-leading) or 0 (S-leading)"
    df['root_correct'] = (df['choice'] == df['BM2']).astype(int) 

    "Map to four patterns"
    def _pattern(row):
        if row['root_correct'] == 1 and row['pred_correct'] == 1: return 'CC'
        if row['root_correct'] == 1 and row['pred_correct'] == 0: return 'CM'
        if row['root_correct'] == 0 and row['pred_correct'] == 1: return 'MC'
        return 'MM'
    df['pattern'] = df.apply(_pattern, axis=1)

    "Counts by quadruplet"
    desired_order = ['SMSM', 'MSMS', 'SMSS', 'MSMM', 'SSSM', 'MMMS', 'SSSS', 'MMMM']
    counts = (df.groupby('quadruplet')['pattern']
                .value_counts()
                .unstack(fill_value=0)
                .reindex(desired_order, fill_value=0))
    for col in ['CC','CM','MC','MM']:
        if col not in counts.columns:
            counts[col] = 0  # ensure all columns present

    counts['N'] = counts[['CC','CM','MC','MM']].sum(axis=1).astype(int)

    "Ratios (six-decimal rounding)"
    ratios = counts[['CC','CM','MC','MM']].div(counts['N'].replace(0, np.nan), axis=0).fillna(0.0)
    ratios = ratios.round(round_to)
    ratios.columns = [c + '/N' for c in ratios.columns]

    "Tidy printout prioritizing ratios and N"
    display_df = pd.concat([ratios, counts['N']], axis=1)

    "Interpretive table (same numbers under semantic labels)"
    def _leading_action(choice):
        return 'Stay' if choice[0] == 'S' else 'Move'  # first letter is BM2 (same as NM2)

    interpret_df = pd.DataFrame({
        'Root-optimal action': [_leading_action(choice) for choice in display_df.index],
        'Coherent high-order ToM (CC)': ratios['CC/N'],
        'Local drop at P2 (CM)':        ratios['CM/N'],
        'Late correction (MC)':         ratios['MC/N'],
        'Coherent low-order ToM (MM)':  ratios['MM/N'],
        'N':                            counts['N'].astype(int),
    }, index=display_df.index)

    return display_df, counts, interpret_df


def prediction_consistency_compact(analyzed_df: pd.DataFrame, round_to: int = 6) -> None:
    """
    Prints two compact tables for the appendix:
      • Move-leading quadruplets: CC/N and CM/N (+ N)
      • Stay-leading quadruplets: MM/N and MC/N (+ N)
    Also prints an overall coherence rate: (CC+MM)/N across all quads.
    """
    "Reuse existing computation"
    ratios_df, counts_df, _ = prediction_consistency_by_quadruplet(analyzed_df, round_to=round_to)

    "Identify move-leading vs stay-leading from the quadruplet tag (first letter = BM2 = NM2)"
    move_leading = ['MSMS','MSMM','MMMS','MMMM']
    stay_leading = ['SMSM','SMSS','SSSM','SSSS']

    "Move-leading panel: CC/N and CM/N"
    move = ratios_df.loc[ratios_df.index.isin(move_leading), ['CC/N','CM/N']].copy().round(round_to)
    move['N'] = counts_df.loc[move.index, 'N'].astype(int)

    "Stay-leading panel: MM/N and MC/N"
    stay = ratios_df.loc[ratios_df.index.isin(stay_leading), ['MM/N','MC/N']].copy().round(round_to)
    stay['N'] = counts_df.loc[stay.index, 'N'].astype(int)

    print("Choice–Prediction Consistency (compact; Move-leading quadruplets):")
    print(move.to_string())

    print("\nChoice–Prediction Consistency (compact; Stay-leading quadruplets):")
    print(stay.to_string())

    print("\nCC = correct-correct,  CM = correct-mistaken,")
    print("MC = mistaken-correct, MM = mistaken-mistaken")

    "Overall coherence across all quadruplets"
    total_cc = counts_df.get('CC', pd.Series(0, index=counts_df.index)).sum()
    total_mm = counts_df.get('MM', pd.Series(0, index=counts_df.index)).sum()
    total_n  = counts_df['N'].sum()
    coherence = (total_cc + total_mm) / total_n if total_n else np.nan
    print(f"\nOverall coherence across all quadruplets: {(coherence*100):.2f}% "
          f"(coherent trials = CC + MM; total = {int(total_n)})\n")


"""
===================================================================================
=============================== VISUALIZING THE DATA ==============================
===================================================================================
"""
def tom_error_rate_violin_chart(erates_individual_groups: pd.DataFrame, fig_lay: dict = fig_lay,
    export_fig: bool = True, out_name: str = "Error_Rates_by_Game_Group_Violin.html", as_box: bool = False) -> go.Figure:
    """
    Violin (or box) plot with jittered individual data points, one distribution per game group.
    X = game group (G1..G4 with intuitive labels), Y = participant error rate for that group.
    """
    n_participants = len(list(set(erates_individual_groups['participant_number'])))

    "Desired group order and display names"
    order = ['G1', 'G2', 'G3', 'G4']
    hues = [120, 80, 40, 0]
    display = {
        'G1': 'Game Group 1:<br>Both models expect<br>high error rates',
        'G2': 'Game Group 2:<br>Only BM expects<br>a high error rate',
        'G3': 'Game Group 3:<br>Only NM expects<br>a high error rate',
        'G4': 'Game Group 4:<br>Neither model expects<br>high error rates'
    }

    fig = go.Figure()

    for idx, group_id in enumerate(order):
        sub = erates_individual_groups[erates_individual_groups['group_id'] == group_id].copy()
        "Hovertext includes counts per participant"
        custom = np.stack(
            [sub['n_correct'].values, sub['n_mistaken'].values, sub['n_abdicated'].values],
            axis=1
        ) if len(sub) else np.empty((0, 3))

        fig.add_trace(
            go.Violin(
                x=[display[group_id]] * len(sub),
                y=sub['error_rate'],
                name=display[group_id],
                box=dict(visible=True, line=dict(color=f"hsla({hues[idx]}, 100%, 45%, 1.0)", 
                                                 width=3)) if not as_box else dict(visible=False),
                marker=dict(color=f"hsla({hues[idx]}, 100%, 50%, 0.8)", 
                            line=dict(color=f"hsla({hues[idx]}, 100%, 15%, 1.0)")),
                meanline=dict(visible=True), points='all', pointpos=0.0, jitter=0.38,
                scalemode='count', customdata=custom,
                hovertemplate=(
                    "%{x}<br>"
                    "Error rate: %{y:.3f}<br>"
                    "Correct: %{customdata[0]}<br>"
                    "Mistaken: %{customdata[1]}<br>"
                    "Abdicated: %{customdata[2]}<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title="Participant Error Rates by Game Group",
        titlefont_size=fig_lay.get("titlefont_size", 48),
        template=fig_lay.get("template", "plotly_white"),
        title_x=fig_lay.get("title_x", 0.5),
        title_y=fig_lay.get("title_y", 0.96),
        font=fig_lay.get("font", dict(family="Calibri", color="black", size=24)),
        violingap=0.08,
        violingroupgap=0.12,
        showlegend=False,
        margin=dict(l=220, r=220, t=120, b=120)
    )
    fig.update_xaxes(
        title="Game Groups (Classified by Both Model's Expected Error Rates)",
        **fig_lay.get("xaxis", {})
    )
    fig.update_yaxes(
        title=f"Mean Error Rate Across {n_participants} Participants",
        range=[0.0, 1.01],
        **fig_lay.get("yaxis", {})
    )

    if export_fig:
        if as_box and "Violin" in out_name:
            out_name = out_name.replace('Violin', 'Boxplot')
        out_path = os.path.join(file_path_figures, out_name)
        print(f"Saved violin/box figure → {out_path.replace(str(ROOT), '...')}")
        fig.write_html(out_path), print("")
    else:
        fig.show()

    return fig


def timecourse_individual_errors(analyzed_df: pd.DataFrame, n_groups: int = 4, include_abdicated: bool = True, 
                                 recreate_csv: bool = False, file_name: str = "Timecourse_Individual_Errors_ToM+.csv") -> pd.DataFrame:
    """
    Organizes participant data into a format suitable for visualization of error rates over time.

    Arguments:
        • analyzed_df: pd.DataFrame; Dataframe containing individual participant responses for each round.
        • n_groups: int; Number of round groups to divide the 32 trials into (e.g., 4 groups of 8 trials).
        • file_name: str; The name of the CSV file that the data will be saved to.

    Notes:
        • Assumes that there are 32 trials in the experiment.
        • Returns a dataframe that includes participant-wise statistics for each round group.

    Returns:
        • pd.DataFrame; Contains columns for participant_number, round_group, correct_rate, n_correct, n_errors_1,
          n_errors_2, and n_abdicated.
    """
    "Input validation"
    for input_type in [('analyzed_df', analyzed_df, pd.DataFrame), ('n_groups', n_groups, int)]:
        if not isinstance(input_type[1], input_type[2]):
            raise TypeError(f"{input_type[0]} must be of type {input_type[2]}, not {type(input_type[1])}!")

    valid_n_groups = [1, 2, 4, 8, 16, 32]
    if n_groups not in valid_n_groups:
        raise ValueError(f"n_groups must be one of the following numbers {valid_n_groups}, not {n_groups}!")

    "Adding n_groups to the file name."
    file_name = file_name.replace('.csv', '')
    file_name += f'_{n_groups}groups.csv'

    "If the dataframe already exists, get it rather than recreate it."
    if not recreate_csv:
        full_path = os.path.join(file_path_clean, file_name)
        if os.path.exists(full_path):
            results_df = pd.read_csv(full_path)    
            if 'Unnamed: 0' in results_df.columns:
                del results_df['Unnamed: 0']

            return results_df

    def group_label(round_number: int, n_groups: int = n_groups) -> int:
        """
        Assigns a group label to each round based on the specified number of groups.

        Arguments:
            • round_number: int; The round number (0-31).
            • n_groups: int; Number of groups to divide rounds into.

        Returns:
            • int; Group label for the given round number.
        """
        group_size = 32 // n_groups
        return round_number // group_size

    def error_type(choice: str, quadruplet: str, model: str = 'NM') -> str:
        """
        Categorizes the response as 'correct', 'error 1', 'error 2', or 'abdicated'.

        Arguments:
            • choice: str; The choice made ('stay', 'move', or 'abdicated').
            • quadruplet: str; The payoff quadruplet for the round.

        Returns:
            • str; The type of response.
        """
        if model == 'BM':
            idx1, idx2 = 0, 1
        else:
            idx1, idx2 = 2, 3

        diagnostic = quadruplet[idx1] != quadruplet[idx2]
        higher_order_choice = {'S': 'stay', 'M': 'move'}.get(quadruplet[idx1], 'none')
        if choice in ['stay', 'move']:
            if choice == higher_order_choice:
                return 'correct'
            else:
                if diagnostic:
                    "Mistakes"
                    return 'error 1'
                else:
                    "Really bad mistakes"
                    return 'error 2'
        else:
            return 'abdicated'

    "Assigning group labels to each round"
    analyzed_df['round_group'] = analyzed_df['round'].apply(
        lambda round_num: group_label(round_num, n_groups))
    "Assigning response type to each row"
    analyzed_df['error_type'] = analyzed_df.apply(
        lambda row: error_type(row['choice'], row['quadruplet'], model='BM'), axis=1)

    "Aggregating the data by participant and round group"
    aggregated_data = []
    grouped = analyzed_df.groupby(['participant_number', 'round_group'])

    for (participant, round_group), group in grouped:
        n_correct = np.sum(group['error_type'] == 'correct')
        n_errors_1 = np.sum(group['error_type'] == 'error 1')
        n_errors_2 = np.sum(group['error_type'] == 'error 2') 
        n_abdicated = np.sum(group['error_type'] == 'abdicated')

        total_responses = n_correct + n_errors_1 + n_errors_2
        if include_abdicated: total_responses += n_abdicated
        correct_rate = n_correct / total_responses if total_responses > 0 else 0

        aggregated_data.append({
            'participant_number': participant,
            'round_group': round_group,
            'correct_rate': correct_rate,
            'n_correct': n_correct,
            'n_errors_1': n_errors_1,
            'n_errors_2': n_errors_2,
            'n_abdicated': n_abdicated
        })

    "Creating a DataFrame from the aggregated data"
    result_df = pd.DataFrame(aggregated_data)

    "Save the dataframe to CSV"
    output_file_path = os.path.join(file_path_clean, file_name)
    result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"Timecourse data saved to {output_file_path.replace(str(ROOT), '...')}")

    return result_df


def ridgeline_timecourse(analyzed_df: pd.DataFrame, n_groups: int = 4, export_fig: bool = True, 
                         file_name: str = "Ridgeline_Timecourse_ToM+.html") -> None:
    """
    Generates a ridgeline plot to visualize the correct rate across the course of the experiment

    Arguments:
        • analyzed_df: pd.DataFrame; Dataframe containing individual participant responses for each round.
        • n_groups: int; Number of round groups to divide the 32 trials into (e.g., 4 groups of 8 trials).
        • export_fig: bool; If True, saves the figure as HTML, otherwise shows it.
        • file_name: str; The name of the file to save the figure as HTML.

    Returns:
        • None: The function generates a figure and either saves or displays it.
    """
    "Adding n_groups to the file name."
    bright_mode = 'dark' if dark_mode else 'light'
    file_name = file_name.replace('.html', '')
    file_name += f'_{n_groups}groups_{bright_mode}.html'

    "Create or extract dataframe"
    df_timecourse = timecourse_individual_errors(analyzed_df=analyzed_df, n_groups=n_groups)

    "Create a ridgeline-like plot using violin traces"
    grouped = df_timecourse.groupby('round_group')
    colors = [f"hsla({int(hue)}, 100%, 50%, 1)" for hue in np.linspace(start=240, stop=270, num=n_groups)]

    n_trials = 32
    n_rounds_per_group = n_trials / n_groups

    fig = go.Figure()

    "Iterate through each round group and add a violin trace"
    for (round_group, group_data), color in zip(reversed(list(grouped)), colors):
        name = f'Rounds {int(round_group * n_rounds_per_group + 1)}-{int((round_group + 1) * n_rounds_per_group)}'
        fig.add_trace(go.Violin(
            name=name, line_color=color, x=group_data['correct_rate'],
            box_visible=True, meanline_visible=True, points=False, 
            width=0.8, side='positive', orientation='h'
        ))

    "Update layout and axes for better visual appearance"
    fig.update_yaxes(title_text="Round Number", title_font=fig_lay['yaxis']['title_font'], showgrid=False)
    fig.update_xaxes(title_text="Correct Rate", title_font=fig_lay['xaxis']['title_font'], showgrid=True, 
                     zeroline=False, range=[-0.1, 1.1], tickvals=[0, .25, 0.5, .75, 1])
    
    fig.update_layout(
        title='Correct Rates by Round Order', titlefont_size=fig_lay['titlefont_size'],
        title_x=fig_lay['title_x'], title_y=fig_lay['title_y'], template=fig_lay['template'],
        font=fig_lay['font'], violingap=0, violingroupgap=0.1, showlegend=False, 
        margin=dict(l=560, r=560, t=120, b=100), 
    )

    "Save or show the figure"
    fig.write_html(os.path.join(file_path_figures, file_name)) if export_fig else fig.show()


def plot_bm_nm_error_ratio_histogram(analyzed_df: dict | None = None, min_errors: int = 6, bin_size=0.1, export_fig: bool = True, 
                                     print_: bool = True, file_name: str = "BM_NM_ErrorRatio_Histogram.html") -> go.Figure:
    """
    Histogram of BM/(BM+NM) per participant with a vertical midline at 0.5.
        • X-axis: Error Ratio: BM / (BM + NM)  (range [0, 1])
        • Y-axis: Number of Participants (N = <included>; Min Errors ≥ <min_errors>)
        • No legend; y-ticks are integers.
    """

    def participant_error_ratio_df(analyzed_df: dict | None = None, 
                                   min_errors: int = 6) -> pd.DataFrame:
        """
        Build a per-participant table of BM vs NM error counts and the simple ratio:
            bm_share = BM_errors / (BM_errors + NM_errors)

        Notes:
            • Uses only *controversy* games so that every error lands in exactly one bucket:
                - BM-only diagnostic (⊥⊤)  → BM bucket
                - NM-only diagnostic (⊤⊥)  → NM bucket
            • Excludes abdications at the root via your cleaned_df() preprocessing.
            • Filters to participants with total errors >= min_errors across those two groups.
        """
        if analyzed_df is None:
            analyzed_df = cleaned_df(recreate_csv=False, print_=False)
        df = analyzed_df['df'].copy()
        opt_col = 'BM2' 

        "Keep only controversy groups (BM-only diagnostic ⊥⊤ and NM-only diagnostic ⊤⊥)"
        df = df[df['mistake'].isin(['⊥⊤', '⊤⊥'])].copy()

        "Valid (non‑abdicated) choices at the root"
        df['is_valid'] = df['choice'].isin(['stay', 'move'])

        "An error is a valid choice that differs from the optimal 2nd‑order choice"
        df['is_error'] = df['is_valid'] & (df['choice'] != df[opt_col])

        "Count BM vs NM errors per participant"
        grouped = df.groupby('participant_number')
        tally = grouped.apply(
            lambda g: pd.Series({
                'bm_err': int(((g['mistake'] == '⊥⊤') & g['is_error']).sum()),
                'nm_err': int(((g['mistake'] == '⊤⊥') & g['is_error']).sum())
            })
        ).reset_index()

        tally['total_err'] = tally['bm_err'] + tally['nm_err']
        keep = tally['total_err'] >= min_errors
        tally = tally.loc[keep].copy()

        "Simple BM-share ratio"
        with pd.option_context('mode.chained_assignment', None):
            tally['bm_share'] = tally['bm_err'] / (tally['bm_err'] + tally['nm_err'])

        return tally

    "Build per-participant table"
    tally = participant_error_ratio_df(analyzed_df=analyzed_df, min_errors=min_errors)

    n_included = len(tally)
    n_nm_lean = int((tally['bm_share'] < 0.5).sum())
    n_bm_lean = int((tally['bm_share'] > 0.5).sum())
    n_ties    = int((tally['bm_share'] == 0.5).sum())

    "Shifted bins so that one centers on 0.5"
    bins = np.round(np.arange(0, 1 + bin_size, bin_size) - bin_size/2, decimals=4).tolist()
    "Clamp edges to [0,1]"
    bins[0], bins[-1] = 0.0, 1.0
    if bins[-1] - bins[-2] > bins[1] - bins[0]:
        bins = [bins[0]] + bins[2:]

    line_color = 'hsla(120, 100%, 20%, 0.9)'

    "Figure scaffolding"
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=tally['bm_share'], 
        xbins=dict(start=bins[0] + bin_size/2, end=bins[-1], size=bin_size),
        texttemplate="%{y}", showlegend=False, textfont_color='black', textfont_size=28, 
        marker=dict(color='hsla(120, 100%, 50%, 0.9)', line=dict(width=3, color=line_color)),
        hovertemplate="Ratio: %{x:.2f}<br>Count: %{y}<extra></extra>"
    ))

    "Vertical reference line at 0.5"
    fig.add_shape(
        type="line",
        x0=0.5, x1=0.5, y0=0, y1=1, yref="paper",
        line=dict(width=6, dash="dot", color=line_color)
    )

    hist, bin_edges = np.histogram(tally['bm_share'], bins=bins)
    max_bin_count = hist.max()

    "Axis titles & ticks"
    upper_tick = max(2, int(np.ceil(max_bin_count / 2.) * 2))
    tickvals = list(range(2, upper_tick + 1, 2))
    ticktext = [str(tv) for tv in tickvals]

    xaxis = dict(title="Error Ratio: BM / (BM + NM)", range=[0, 1], tick0=0.0, dtick=0.1)
    yaxis = dict(
        title=f"Number of Participants (N = {n_included}; Min Errors ≥ {min_errors})",
        tickmode="array", tickvals=tickvals, ticktext=ticktext
    )

    layout_kwargs = dict(
        title="Participants by BM-vs-NM Error Ratio",
        template=fig_lay.get("template", "plotly_white") if 'fig_lay' in globals() else "plotly_white",
        xaxis=fig_lay.get("xaxis", {}) | xaxis if 'fig_lay' in globals() else xaxis,
        yaxis=fig_lay.get("yaxis", {}) | yaxis if 'fig_lay' in globals() else yaxis,
        font=fig_lay.get("font", dict(family="Calibri", size=24, color="black")) 
            if 'fig_lay' in globals() else dict(family="Calibri", size=24, color="black"),
        title_x=fig_lay.get("title_x", 0.5) if 'fig_lay' in globals() else 0.5,
        title_y=fig_lay.get("title_y", 0.96) if 'fig_lay' in globals() else 0.96,
        titlefont_size=fig_lay.get("titlefont_size", 48) if 'fig_lay' in globals() else 48,
        margin=dict(l=120, r=120, t=120, b=120)
    )
    fig.update_layout(**layout_kwargs)

    if export_fig:
        out_dir = file_path_figures if 'file_path_figures' in globals() else "."
        safe_name = "".join(ch for ch in file_name if ch not in " ,<>:/\\|?*'[]")
        fig.write_html(os.path.join(out_dir, safe_name))

    if print_:
        print("BM/(BM+NM) participant classification (disagreement games only):")
        print(f"  Min errors ≥ {min_errors} \n  N included = {n_included}")
        print(f"  NM-leaning (ratio < 0.5): {n_nm_lean}")
        print(f"  BM-leaning (ratio > 0.5): {n_bm_lean}")
        print(f"  Ties (ratio = 0.5):       {n_ties}")
        print("")

    return fig


def plot_rt_by_group_violin(rt_group_df: pd.DataFrame, export_fig: bool = True, out_name: str = "RT_by_Game_Group_Violin.html") -> go.Figure:
    """
    Violin (with box + jitter) of participant-level median RTs by game group (G1..G4).
    """
    order = ['G1', 'G2', 'G3', 'G4']
    hues = [120, 80, 40, 0]
    display = {
        'G1': 'Game Group 1:<br>Both models expect<br>high error rates',
        'G2': 'Game Group 2:<br>Only BM expects<br>a high error rate',
        'G3': 'Game Group 3:<br>Only NM expects<br>a high error rate',
        'G4': 'Game Group 4:<br>Neither model expects<br>high error rates'
    }

    fig = go.Figure()
    for idx, gid in enumerate(order):
        sub = rt_group_df[rt_group_df['group_id'] == gid].copy()
        custom = sub['n_trials'].to_numpy().reshape(-1, 1)

        fig.add_trace(go.Violin(
            x=[display[gid]]*len(sub), y=sub['median_choice_rt'], name=display[gid],
            box=dict(visible=True, line=dict(color=f"hsla({hues[idx]}, 100%, 45%, 1.0)", width=3)),
            meanline=dict(visible=True), points='all', pointpos=0.0, jitter=0.38, scalemode='count', customdata=custom,
            marker=dict(color=f"hsla({hues[idx]}, 100%, 50%, 0.9)", line=dict(color=f"hsla({hues[idx]}, 100%, 20%, 0.9)")),
            hovertemplate="Group: %{x}<br>Median RT: %{y:.3f}s<br>Trials: %{customdata[0]}<extra></extra>"
        ))

    fig.update_layout(
        title="Participant Median RTs by Game Group",
        titlefont_size=fig_lay.get("titlefont_size", 48),
        template=fig_lay.get("template", "plotly_white"),
        title_x=fig_lay.get("title_x", 0.5),
        title_y=fig_lay.get("title_y", 0.96),
        font=fig_lay.get("font", dict(family="Calibri", color="black", size=24)),
        violingap=0.08, violingroupgap=0.12, showlegend=False,
        margin=dict(l=220, r=220, t=120, b=120)
    )
    fig.update_xaxes(title="Game Groups (Classified by Both Model's Expected Error Rates)")
    fig.update_yaxes(title="Median root-node RT")

    if export_fig:
        out_path = os.path.join(file_path_figures, out_name)
        fig.write_html(out_path)
        print(f"Saved RT-by-group violin → {out_path}")
    else:
        fig.show()
    return fig


"""
===================================================================================
================================ ANALYZING THE DATA ===============================
===================================================================================
"""
def timecourse_analysis(analyzed_df: pd.DataFrame, n_groups: int = 4, include_abdicated: bool = True, 
                        recreate_csv: bool = False, file_name: str = "Timecourse_Individual_Errors_ToM+.csv") -> None:
    """
    Analyzes the data on error rates over time and prints results to the terminal.

    Arguments:
        • analyzed_df: pd.DataFrame; Dataframe containing individual participant responses for each round.
        • n_groups: int; Number of round groups to divide the 32 trials into (e.g., 4 groups of 8 trials).
        • include_abdicated: bool; If True, classifies abdicated responses as errors.
        • recreate_csv: bool; If true, regenerates the error rates dataframe.
        • file_name: str; The name of the error rates CSV file.

    Returns:
        • None: Prints analysis to terminal.
    """
    df = timecourse_individual_errors(analyzed_df=analyzed_df, n_groups = n_groups, 
            include_abdicated = include_abdicated, recreate_csv = recreate_csv, file_name = file_name)

    "Pivot the data to wide format"
    df_pivot = df.pivot(index='participant_number', columns='round_group', values='correct_rate')

    "Ensure there are no missing values"
    df_pivot = df_pivot.dropna()

    "Rename the columns for clarity"
    df_pivot.columns = ['Block1', 'Block2', 'Block3', 'Block4']

    "Extract the correct rates for each block"
    block1 = df_pivot['Block1']
    block2 = df_pivot['Block2']
    block3 = df_pivot['Block3']
    block4 = df_pivot['Block4']

    "Calculate mean correct rate for each block"
    mean_block1 = block1.mean()
    mean_block2 = block2.mean()
    mean_block3 = block3.mean()
    mean_block4 = block4.mean()

    print("Mean correct rate for Block 1:", mean_block1)
    print("Mean correct rate for Block 2:", mean_block2)
    print("Mean correct rate for Block 3:", mean_block3)
    print("Mean correct rate for Block 4:", mean_block4)
    print("")

    "Perform the Friedman Test"
    statistic, p_value = friedmanchisquare(block1, block2, block3, block4)

    print(f'Friedman Test Statistic: {statistic:.3f}, p-value: {p_value:.3f}')

    "Calculate Kendall's W"
    k = 4  # Number of conditions (blocks)
    N = len(df_pivot)  # Number of participants
    W = statistic / (N * (k - 1))
    print(f"Kendall's W: {W:.3f}")

    "Define a function for pairwise comparisons"
    def pairwise_wilcoxon(data1, data2):
        stat, p = wilcoxon(data1, data2)
        return stat, p

    "List of block pairs"
    block_pairs = [('Block1', 'Block2'),
                ('Block1', 'Block3'),
                ('Block1', 'Block4'),
                ('Block2', 'Block3'),
                ('Block2', 'Block4'),
                ('Block3', 'Block4')]

    "Perform pairwise comparisons"
    results = []
    for pair in block_pairs:
        stat, p = pairwise_wilcoxon(df_pivot[pair[0]], df_pivot[pair[1]])
        "Adjust p-value using Bonferroni correction"
        p_adjusted = p * len(block_pairs)
        p_adjusted = min(p_adjusted, 1.0)  #Cap at 1.0
        results.append({
            'Comparison': f'{pair[0]} vs {pair[1]}',
            'Statistic': stat, 'p-value': p,
            'p-value adjusted': p_adjusted
        })

    "Convert results to DataFrame"
    results_df = pd.DataFrame(results)

    "Display the results"
    print(results_df)
    print("")


def chi_square_test(error_results_dataframe: pd.DataFrame = error_results_df(), test_label: str = "") -> None:
    """
    Runs Chi-Square tests on the error results dataframe and prints the results.

    Tests included:
        • (1&3) vs (2&4): NM prediction
        • (1&2) vs (3&4): BM predictions (ignore-D vs ignore-C)
        • 1 vs 3: NM internal consistency check
        • 2 vs 4: NM internal consistency check
        • Abdications across groups
        • Omnibus 2×4 test: uniformity of error rates across all groups

    Arguments:
        • error_results_dataframe : pd.DataFrame
            Dataframe with columns ['game_type', 'correct', 'mistaken', 'abdicated'].
        • test_label: str: Label of test for clarity while printing.
    """
    data_by_group = {}
    for group_label, game_type_symbol in [
        ('group_1', '⊥⊥'),
        ('group_2', '⊥⊤'),
        ('group_3', '⊤⊥'),
        ('group_4', '⊤⊤')
    ]:
        data_by_group[group_label] = {}
        for response_type in ['correct', 'mistaken', 'abdicated']:
            data_by_group[group_label][response_type] = (
                error_results_dataframe.loc[
                    error_results_dataframe['game_type'] == game_type_symbol,
                    response_type
                ].values[0]
            )

    "Print the raw counts for transparency"
    print(f"Chi-square tests {test_label}:")
    pp.pprint(data_by_group), print("")

    contingency_tables = {
        '(1&3)v(2&4)': np.array([
            [data_by_group['group_1']['correct'] + data_by_group['group_3']['correct'],
             data_by_group['group_1']['mistaken'] + data_by_group['group_3']['mistaken']],
            [data_by_group['group_2']['correct'] + data_by_group['group_4']['correct'],
             data_by_group['group_2']['mistaken'] + data_by_group['group_4']['mistaken']]
        ]),
        '(1&2)v(3&4)': np.array([
            [data_by_group['group_1']['correct'] + data_by_group['group_2']['correct'],
             data_by_group['group_1']['mistaken'] + data_by_group['group_2']['mistaken']],
            [data_by_group['group_3']['correct'] + data_by_group['group_4']['correct'],
             data_by_group['group_3']['mistaken'] + data_by_group['group_4']['mistaken']]
        ]),
        '1v3': np.array([
            [data_by_group['group_1']['correct'], data_by_group['group_1']['mistaken']],
            [data_by_group['group_3']['correct'], data_by_group['group_3']['mistaken']]
        ]),
        '2v4': np.array([
            [data_by_group['group_2']['correct'], data_by_group['group_2']['mistaken']],
            [data_by_group['group_4']['correct'], data_by_group['group_4']['mistaken']]
        ]),
        'abdicated': np.array([
            [data_by_group['group_1']['correct'] + data_by_group['group_1']['mistaken'],
             data_by_group['group_1']['abdicated']],
            [data_by_group['group_2']['correct'] + data_by_group['group_2']['mistaken'],
             data_by_group['group_2']['abdicated']],
            [data_by_group['group_3']['correct'] + data_by_group['group_3']['mistaken'],
             data_by_group['group_3']['abdicated']],
            [data_by_group['group_4']['correct'] + data_by_group['group_4']['mistaken'],
             data_by_group['group_4']['abdicated']]
        ]),
        'omnibus_uniformity': np.array([
            [data_by_group['group_1']['mistaken'], data_by_group['group_2']['mistaken'],
             data_by_group['group_3']['mistaken'], data_by_group['group_4']['mistaken']],
            [data_by_group['group_1']['correct'], data_by_group['group_2']['correct'],
             data_by_group['group_3']['correct'], data_by_group['group_4']['correct']]
        ])
    }

    for comparison_label, contingency_table in contingency_tables.items():
        chi2_statistic, p_value, degrees_of_freedom, expected_frequencies = chi2_contingency(contingency_table)
        sample_size = np.sum(contingency_table)
        num_rows, num_columns = contingency_table.shape
        cramer_v = np.sqrt(chi2_statistic / (sample_size * (min(num_columns - 1, num_rows - 1))))
        print(
            f"Comparison {comparison_label}: "
            f"\nChi-square Stat: {chi2_statistic:.3f}, p-value: {p_value:.3e}, "
            f"\ndof: {degrees_of_freedom}, Cramér's V: {cramer_v:.4f}, N = {int(sample_size)}, "
            f"\nExpected frequencies: \n{np.round(expected_frequencies, 2)}\n"
        )


def error_distribution_table(eresults_df: pd.DataFrame, round_to: int = 6) -> pd.DataFrame:
    """
    Prints the values for the table: Error Count Across Four Game Categories 
    """
    "Keep only the 4 groups (not quadruplets)"
    groups = eresults_df[eresults_df['game_type'].isin(['⊥⊥', '⊥⊤', '⊤⊥', '⊤⊤'])].copy()

    "Build a nice table in the order G1..G4"
    order = ['⊥⊥', '⊥⊤', '⊤⊥', '⊤⊤']
    groups = groups.set_index('game_type').loc[order].reset_index()

    table = groups[['game_type', 'correct', 'mistaken', 'abdicated', 'total', 
                    'error_rate', 'error_ci_low', 'error_ci_high']].copy()

    "Round rates/CIs"
    for col in ['error_rate', 'error_ci_low', 'error_ci_high']:
        table[col] = table[col].astype(float).round(round_to)

    print("Error rates by game group:")
    print(table.to_string(index=False))

    "Optionally: Show how mistakes distribute across groups"
    total_mistakes = groups['mistaken'].sum()
    if total_mistakes > 0:
        shares = (groups[['game_type', 'mistaken']].assign(mistake_share = lambda d: d['mistaken'] / total_mistakes)
                  .drop(columns=['mistaken']))
        shares['mistake_share'] = shares['mistake_share'].round(round_to)
        print("\nProportion of all mistakes by game group:")
        print(shares.to_string(index=False))

    print("")
    return table


def robustness_check_error_groups(max_abdications: int = 31, save_variants: bool = True, round_to: int = 6, print_: bool = True) -> None:
    """
    Checks if excluding participants significantly changes results. Prints side-by-side group 
    error rates with and without exclusions, then re-runs the χ² contrasts for each version. 

    Arguments:
        • max_abdications: int; Maximum number of abdications before participant is excluded
        • save_variants: bool; If true, saves the resulting dataframes. 
        • round_to: int; Rounding digits for pretty printing.
    """
    def error_results_exclusions(max_abdications: int = 31, save_suffix: str | None = None) -> pd.DataFrame:
        """
        Builds an ERates dataframe using a specific exclusion threshold and 
        (optionally) saves it under a suffix (e.g., '_INCL_All' or '_EXCL_gt9').
        """
        analyzed = cleaned_df(recreate_csv=False, print_=False, max_abdications=max_abdications)
        results = count_mistakes(analyzed, confidence_intervals=True, print_=False)
        df = pd.DataFrame(results).transpose().reset_index().rename(columns={'index': 'game_type'})

        if save_suffix:
            out = os.path.join(file_path_clean, f"Morality_Game_Study_Results_Higher_ToM_ERates{save_suffix}.csv")
            df.to_csv(out, index=False, encoding='utf-8-sig')
        return df

    df_excl = error_results_exclusions(max_abdications=max_abdications,
                                       save_suffix="_EXCL_gt9" if save_variants else None)
    df_incl = error_results_exclusions(max_abdications=999,
                                       save_suffix="_INCL_All" if save_variants else None)

    keep = ['game_type', 'correct', 'mistaken', 'abdicated', 'total', 'error_rate', 'error_ci_low', 'error_ci_high']
    a = df_excl[df_excl['game_type'].isin(['⊥⊥','⊥⊤','⊤⊥','⊤⊤'])][keep].copy().set_index('game_type')
    b = df_incl[df_incl['game_type'].isin(['⊥⊥','⊥⊤','⊤⊥','⊤⊤'])][keep].copy().set_index('game_type')

    "Round rates/CIs for display"
    for col in ['error_rate','error_ci_low','error_ci_high']:
        a[col] = a[col].astype(float).round(round_to)
        b[col] = b[col].astype(float).round(round_to)

    merged = a.join(b, lsuffix='_EXCL', rsuffix='_INCL')
    merged['Δ_error_rate'] = (merged['error_rate_EXCL'] - merged['error_rate_INCL']).round(round_to)

    if print_:
        print("Robustness check (group error rates with vs. without exclusions):")
        print(merged[['error_rate_INCL','error_rate_EXCL','Δ_error_rate',
                    'correct_INCL','mistaken_INCL','abdicated_INCL','total_INCL',
                    'correct_EXCL','mistaken_EXCL','abdicated_EXCL','total_EXCL']].to_string())

        "χ² on each version"
        chi_square_test(df_incl, test_label="(including all participants)")
        chi_square_test(df_excl, test_label=f"(excluding > {max_abdications} abdications)")


def robustness_abdications_as_errors(error_results_dataframe: pd.DataFrame | None = None, print_: bool = True) -> pd.DataFrame:
    """
    Recomputes error rates when abdications are treated as errors 
    and reruns the planned (Group 1 & 3) vs (Group 2 & 4) contrast.
    
    Arguments:
        • error_results_dataframe: DataFrame with columns
            ['game_type', 'correct', 'mistaken', 'abdicated', 'total', ...]
            where 'game_type' holds the quadruplet or group identifier.

    Returns:
        • pd.DataFrame; DataFrame with error rates excluding vs including abdications.
    """
    if error_results_dataframe is None:
        error_results_dataframe = error_results_df(recreate_csv=False).copy()

    results_df = error_results_dataframe.copy()

    "Compute error rates including abdications as errors"
    results_df['errors_including_abd'] = results_df['mistaken'] + results_df['abdicated']
    results_df['error_rate_excluding_abd'] = results_df['mistaken'] / (results_df['correct'] + results_df['mistaken'])
    results_df['error_rate_including_abd'] = results_df['errors_including_abd'] / results_df['total']
    results_df['delta_error_rate'] = results_df['error_rate_including_abd'] - results_df['error_rate_excluding_abd']

    if print_:
        print("Robustness check — counting abdications as errors")
        print(results_df[['game_type', 'error_rate_excluding_abd', 'error_rate_including_abd', 
                        'delta_error_rate', 'correct', 'mistaken', 'abdicated', 'total']]
            .to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    "Replace these sets with the actual mapping you use elsewhere"
    group_1_types = {'SMSM','MSMS'}   # both diagnostic
    group_2_types = {'SMSS','MSMM'}   # BM-only diagnostic
    group_3_types = {'SSSM','MMMS'}   # NM-only diagnostic
    group_4_types = {'SSSS','MMMM'}   # neither diagnostic

    def assign_group(game_type: str) -> str:
        if game_type in group_1_types:
            return "group_1"
        elif game_type in group_2_types:
            return "group_2"
        elif game_type in group_3_types:
            return "group_3"
        elif game_type in group_4_types:
            return "group_4"
        else:
            return "unknown"

    results_df['group'] = results_df['game_type'].apply(assign_group)

    "Collapse counts by group"
    grouped = results_df.groupby('group').agg({
        'correct':'sum',
        'errors_including_abd':'sum',
        'total':'sum'
    }).reset_index()

    "Build contingency table: (Group1+Group3) vs (Group2+Group4)"
    correct_13 = int(grouped.loc[grouped['group'].isin(['group_1','group_3']), 'correct'].sum())
    errors_13 = int(grouped.loc[grouped['group'].isin(['group_1','group_3']), 'errors_including_abd'].sum())

    correct_24 = int(grouped.loc[grouped['group'].isin(['group_2','group_4']), 'correct'].sum())
    errors_24 = int(grouped.loc[grouped['group'].isin(['group_2','group_4']), 'errors_including_abd'].sum())

    contingency_table = np.array([[correct_13, errors_13],
                                  [correct_24, errors_24]])

    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    n_total = contingency_table.sum()
    cramers_v = math.sqrt(chi2_stat / (n_total * min(contingency_table.shape)-1))

    if print_:
        print(f"\n(1&3) vs (2&4) INCLUDING abdications as errors:"
            f" χ² = {chi2_stat:.3f}, p = {p_value:.3g}, dof = {dof}, Cramér's V = {cramers_v:.4f}, N = {n_total}")
        print("Expected frequencies:\n", expected), print("")

    return results_df


"""
===================================================================================
================================= RUNNING THE CODE ================================
===================================================================================
"""

if __name__ == "__main__":
    recreate_csv = False
    max_abdications = 31
    print_robustness_checks = False
    run_pref_weight_models = False
    "Set n_iters_per_simulation to 1_000_000 to run the full version."
    n_iters_per_simulation = 1_000

    "Generating the data"
    analyzed_data = cleaned_df(recreate_csv=recreate_csv, print_=True, max_abdications=max_abdications)
    analysis_cols = analyzed_data['analysis_cols']
    response_cols = analyzed_data['response_cols']
    primary_cols = analyzed_data['primary_cols']
    tom_trees_df = analyzed_data['df']
    if recreate_csv:
        error_results_df(recreate_csv=recreate_csv)

    "Individual-level data"
    erates_indiv_groups, erates_indiv_quads = make_erates_individual(analyzed_df=tom_trees_df)
    erates_with_medians = add_median_rts_to_erates(analyzed_df=tom_trees_df, eresults_df=error_results_df(recreate_csv=False))

    "Visualizing the data"
    tom_error_rate_violin_chart(erates_individual_groups=erates_indiv_groups, export_fig=True)
    df_timecourse = timecourse_individual_errors(analyzed_df=tom_trees_df, n_groups=4)
    ridgeline_timecourse(analyzed_df=tom_trees_df, n_groups=4, export_fig=True)

    "Build the per-participant error ratio table & histogram"
    plot_bm_nm_error_ratio_histogram(analyzed_df=cleaned_df(recreate_csv=False),
                                    min_errors=3, bin_size=0.1, export_fig=True)

    "Choice-prediction consistency analysis"
    prediction_consistency_compact(tom_trees_df, round_to=6)

    "Analyzing the data"
    robustness_check_error_groups(max_abdications=max_abdications, save_variants=True, print_=print_robustness_checks)
    robustness_abdications_as_errors(error_results_df(recreate_csv=False), print_=print_robustness_checks)
    error_distribution_table(eresults_df=error_results_df(recreate_csv=False), round_to=6)
    tc_analysis = timecourse_analysis(analyzed_df=tom_trees_df, n_groups=4)
    chi_test = chi_square_test()

    "Reaction time analysis"
    rt_outcome_df = participant_rt_by_outcome(tom_trees_df)
    rt_group_df   = participant_rt_by_group(tom_trees_df)
    rt_statistics(rt_outcome_df, rt_group_df)
    plot_rt_by_group_violin(rt_group_df)

    "Alternative model simulation"
    p_lapse_to_k1_range = [0/6, 1/6, 2/6, 3/6, 4/6, 5/6, 6/6]
    sim_file_name = "Alternative_ToM+_Model_Simulation_Results.csv"
    for mod_types in [
        (False, False, False), 
        (True,  False, False), 
        (False, True,  False), 
        (False, False, True)
    ]:
        if n_iters_per_simulation >= 1:
            simulate_alternative_models(
                n_iters=n_iters_per_simulation, rand_seed=123, 
                print_=True, p_lapse_to_k1_range=p_lapse_to_k1_range,
                export_csv=True, file_name=sim_file_name,
                rand_payoff_models=mod_types[0], 
                rand_switch_models=mod_types[1], 
                noise_prop_model=mod_types[2]
            )

    if run_pref_weight_models:
        preference_weight_p1_range = list(np.linspace(start=-1, stop=1, num=51))
        preference_weight_p2_range = list(np.linspace(start=-1, stop=1, num=51))
        choice_temperature_range = (0.0, 0.1, 0.2, 0.4, 0.8,)
        collapse_to_binary_choice, p_lapse_to_k1 = True, 1.0
        rand_seed, n_iters = None, 100

        simulate_preference_weighting_model(selected_trees = selected_trees, 
                                            out_csv = "Preference_Weighting_Model_Simulation.csv", 
                                            preference_weight_p1_range = preference_weight_p1_range, 
                                            preference_weight_p2_range = preference_weight_p2_range, 
                                            choice_temperature_range = choice_temperature_range, 
                                            collapse_to_binary_choice = collapse_to_binary_choice, 
                                            p_lapse_to_k1 = p_lapse_to_k1, n_iters = n_iters, 
                                            rand_seed = rand_seed, export_csv = True, print_ = True)

        plot_preference_weight_heatmap(grid_csv_name="Preference_Weighting_Model_Simulation.csv", 
                                    df=None, metric="fit", tau_value=0.8, fig_lay=fig_lay, 
                                    preference_weight_p1_range = preference_weight_p1_range, 
                                    preference_weight_p2_range = preference_weight_p2_range, 
                                    choice_temperature_range = choice_temperature_range, 
                                    collapse_to_binary_choice = collapse_to_binary_choice, 
                                    p_lapse_to_k1 = p_lapse_to_k1, rand_seed = rand_seed, 
                                    n_iters = n_iters)
