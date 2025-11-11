"Standardizes and simplifies random sampling from probability density functions."
file_path_inputs,   file_path_outputs = "./Inputs/General", "./Outputs/Exp_Visuals"
import numpy as np, scipy as sp, plotly.graph_objects as go, os, random
from plotly.subplots import make_subplots


def hsla(hue: int, saturation: int = 65, lightness: int = 50, alpha: float = 1.0, return_type: str = "string") -> str | tuple:
    """
    For convenience, hsla() produces a properly formatted hsla string or tuple.

    Arguments:
        • hue: int, angle degrees ⊂ [0°, 360°); Controls the hue, like the angle of on a color wheel.
        • saturation: int ⊂ [0, 100]; Controls the depth or vibrancy of the color.
        • lightness: int ⊂ [0, 100]; Conrols how bright the color is.
        • alpha: float ⊂ [0.0, 1.0]; 0.0 = transparent.  1.0 = opaque.
        • return_type: 'tuple' | 'string'; Determines returned data type.

    Returns: tuple or string (depends on return_type);
        • Tuple example: (180, 65, 50, 1.0)
        • String example: 'hsla(180, 65%, 50%, 1.0)'
    """ 
    if return_type == "tuple": 
        return (
            int((hue*360/(360+1))%360), 
            int(saturation), 
            int(lightness), 
            round(alpha, 2)
            )
    
    elif return_type == "string": 
        return f"hsla({int((hue*360/(360+1))%360)}, {int(saturation)}%, {int(lightness)}%, {round(alpha, 2)})"
    
    else: raise ValueError(f"return_type must be 'tuple' or 'string', not {return_type}.")
    

def edit_file_name(file_name: str, max_file_name_length: int = 110, illegal_characters: str = ".,0<>:/\|?*'[ ]") -> str:
    """
    Edits file name ensuring that it uses legal characters and is of an appropriate length.

    Arguments:
        • file_name: string; The unedited name of your file.
        • max_file_name_length: int; The maximum number of characters in the file name.
        • illegal_characters: string; Removes these characters from the file name.

    Returns:
        • file_name: string; Edited file name
    """
    for character in illegal_characters: 
        file_name = file_name.replace(character, "") 

    if len(str(file_name)) > max_file_name_length: 
        file_name = str(file_name)[:max_file_name_length]  

    return file_name    


def normalize(array: list[float], sumto1: bool = False) -> list[float]:
    """
    Normalizes an array of floating point numbers.

    Arguments:
        • array: list[float]; List of floating point numbers.
        • sumto1: bool;
            - If True, all values in the array will sum to 1.
            - If False, all values will range between 0 and 1.

    Returns:
        • array: list[float]: List of normalized floating point numbers.
    """
    if sumto1:
        total = sum(array)
        for idx in range(len(array)):
            array[idx] = array[idx] / total

    else:
        min_val = min(array)
        max_val = max(array)

        for idx in range(len(array)):
            array[idx] = (array[idx] - min_val) / (max_val - min_val)

    return array


def sample_pdf_linear(low: float = 0.0, high: float = 1.0, slope: float = 0.0, num_samples: int = 1) -> np.ndarray[float]:
    """
    Samples from a distribution created by y = slope * (x - 0.5) + 0.5.

    Adjusting the slope of this distribution is like moving a balance beam with a fulcrum at xy coordinates (0.5, 0.5),
    where a slope of -1 creates a 135° diagonal, 1 creates a 45° diagonal, and 0 creates a uniform distribution.  

    Arguments:
        • low: float (default 0.0, strongly recommended); The leftmost x coordinate; beginning of the interval.
        • high: float (default 1.0, strongly recommended); The rightmost x coordinate; ending of the interval.
        • slope: float (min: -1.0, max: 1.0); Controls the angle or slope of the distribution.
        • num_samples: int > 0; The number of samples in the returned array.

    Returns:
        • samples: np.ndarray[float]; Array of floats producing the the distribution.
    """
    def pdf_linear(xval, slope):
        yval = slope * (xval - 0.5) + 0.5
        return yval / np.sum(yval)

    xval = np.linspace(low, high, 1000)
    yval = pdf_linear(xval, slope)

    samples = np.random.choice(xval, num_samples, p=yval)

    return samples


def normal_sigmoid(xvals: float | list[float] = np.linspace(0, 1, 1000), slope: float = 0.5, 
    midpoint: float = 0.5, constant: float = 5.5452, show: bool = False, export_fig: bool = True) -> np.ndarray[float]:
    """
    S-curve where the slope ranges between 0, when the curve is flat, to 1, when the curve becomes 
    a step function.  This is used to establish the probability that the experiment will continue
    to the next round given number of rounds completed.
    
    Arguments:
        • xvals: float | list[float]; number or list of numbers for completed rounds
        • slope: float; Defines the slope of the S-curve 
        • midpoint: float; Defines the midpoint of the S-curve
        • constant: float; If slope = 0.5, midpoint = 0.5, and constant = 5.5452, then the integral 
            between 0 and 0.5 will be three times larger than the integral between 0.5 and 1.
        • show: boolean; If True, this will generate a Plotly graph.
        • export_fig: boolean; If True, then this will export the graph.

    Notes
        • This can be used for other purposes besides p(continue to next round).
        • Slope should always be positive when using to determine continuation 
            probability, but can be between -1 and 1 otherwise. 
    
    Returns: 
        • yvals: float | list[float]; Probability(ies) of continuing 
            to the next round given the number of rounds completed 
    """
    if midpoint < 0: raise ValueError("Midpoint cannot be negative")
    elif midpoint > 100: print("Warning: midpoint is greater than 100")
    if isinstance(xvals, (np.ndarray, list)) and (midpoint < xvals[0] or midpoint > xvals[-1]):
        raise ValueError("xvals must contain the midpoint.")
    if slope <= -1: raise ValueError("Slope must be positive")
    elif slope > 1: raise ValueError("Slope must be 1 or less.")
    elif slope == 1:
        yvals = np.ones_like(xvals)
        yvals[xvals < midpoint] = 1
        yvals[xvals >= midpoint] = 0
    else: yvals = 1 / (1 + np.exp(constant * (slope / (1 - slope)) * (xvals - midpoint)))

    if show and isinstance(xvals, (np.ndarray, list)):
        "Generating Plotly figure."
        fig = go.Figure()
        rand_color = random.randrange(360)
        fig.add_trace(go.Scatter(x=xvals, y=yvals, name=f"Sigmoid Function", fill='tozeroy', 
            marker=dict(color=hsla(hue=rand_color, saturation=100, lightness=50, alpha=0.6)),
            hovertemplate=f"<b>S-curve</b><br><br>" + "x: %{x:.2f}<br>" + "y: %{y:.2f}<br>"))
        
        fig.update_xaxes(title='X-axis', tickfont=dict(size=20), range=[0, 1], tickvals=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        fig.update_yaxes(title='Y-axis', tickfont=dict(size=20), range=[0, 1], tickvals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        fig.update_layout(title='Normalized Sigmoid Distribution', title_x=0.5, title_y=0.98, titlefont_size=30, 
            hoverlabel=dict(font_size=30), legend=dict(orientation="h"), template="plotly_dark", 
            xaxis=dict(rangeslider=dict(visible=True)),
            font=dict(family='Calibri', size=22), sliders=[
                dict(active=1,
                    currentvalue={"font": {"size": 22}, "prefix": "Slope: ", "visible": True, "xanchor": "right"},
                    pad={"t": 50}, len=0.8, x=0.1, y=-0.25, steps=[dict(method="update",
                        args=[{"y": [1 / (1 + np.exp(constant * (slope_ / (1 - slope_)) * (xvals - midpoint)))]}],
                        label=f"{slope_:.2f}") for slope_ in np.arange(0, 1.01, 0.01)]),
                dict(active=1,
                    currentvalue={"font": {"size": 22}, "prefix": "Midpoint: ", "visible": True, "xanchor": "right" },
                    pad={"t": 50}, len=0.8, x=0.1, y=-0.5, steps=[dict(method="update",
                        args=[{"y": [1 / (1 + np.exp(constant * (slope / (1 - slope)) * (xvals - midpt_)))]}],
                        label=f"{midpt_:.2f}") for midpt_ in np.arange(0, 1.01, 0.01)])])

        file_name = edit_file_name(file_name=f"Sigmoid_Distribution_slope{slope}_midpoint{midpoint}_constant{constant}")
        fig.write_html(os.path.join(file_path_outputs, file_name + ".html")) if export_fig else fig.show()

    return yvals


def sample_pdf_sigmoid(slope: float = 0.5, midpoint: float = 0.5, num_samples: int = 1) -> np.ndarray[float]:
    """
    Randomly samples from an S-curve pdf normalized between y-axis values 0.0 and 1.0.

    Arguments:
        • slope: float (min: -1.0, max: 1.0); Controls the angle or slope of the distribution.
        • midpoint: float (min: 0.0); Midpoint of the distribution, intersects y-axis value 0.5.
        • num_samples: int > 0; The number of samples in the returned array.

    Returns:
        • samples: np.ndarray[float]; Array of floats producing the the distribution.
    """
    xvals = np.linspace(0, 2 * midpoint, 1000)

    yvals = normal_sigmoid(xvals=xvals, slope=slope, midpoint=midpoint)

    samples = np.random.choice(xvals, num_samples, p=yvals / np.sum(yvals))

    return samples


"A dictionary containing the functions and positions of the arguements."
distributions = {
    'linear':     {'function': sample_pdf_linear,    'args': [0, 1, 2, 4]},
    'sigmoid':    {'function': sample_pdf_sigmoid,   'args': [2, 3, 4]},
    'uniform':    {'function': np.random.uniform,    'args': [0, 1, 4]},
    'triangular': {'function': np.random.triangular, 'args': [0, 2, 1, 4]},
    'normal':     {'function': np.random.normal,     'args': [2, 3, 4]},
    'beta':       {'function': np.random.beta,       'args': [2, 3, 4]},
}

distribution_types = list(distributions.keys())


def sample_distribution(distribution_type: str, interval: list[float] = [0.0, 1.0], 
                        coefficients: list[float] = [0.5, 0.5], size: int = 1, print_: bool = False) -> np.ndarray[float]:
    """
    Randomly samples from any of the distributions defined in the distributions dictionary.
   
    This allows any of the distributions to be sampled using just one function and a standard 
    set of arguements.  The goal is to elimenate the need to write contingencies for each type
    of probability density function.  This can simplify the user interface also.    

    Arguments
        • distribution_type: string; The name of a distribution; must be in distribution_types
        • interval: list of two numbers; The low and high values on the x-axis to sample from
        • coefficients: list of two numbers; Coefficients depends on the distribution:
            - linear:     [slope, None]
            - sigmoid:    [slope, midpoint]
            - uniform:    [None,  None]
            - triangular: [mode,  None]
            - normal:     [mean,  variance]
            - beta:       [alpha, beta]
        • size: integer; The desired number of samples
        • print_: boolean; Prints samples if True

    returns: 
        • samples: list of floats; Array of values randomly sampled from the selected distribution.
    """
    if distribution_type not in distributions:
        raise ValueError(f"distribution_type must be one of the following: {distribution_types}")

    if interval[0] == interval[1]: return [interval[0]] * size

    args_lst = [interval[0], interval[1], coefficients[0], coefficients[1], size]

    args = [args_lst[arg] for arg in distributions[distribution_type]['args']]

    samples = distributions[distribution_type]['function'](*args)

    if print_: print(samples)

    return samples


def plot_distribution(distribution_type: str, interval: list[float], coefficients: list[float], 
                      samples: list[float], export_fig: bool = True) -> None:
    """
    Plots the ideal versus sample distribution for a given distribution type.
    Two y-axes are created to visualize the two distributions at a comparable scale.
    
    Arguments:
        • distribution_type: string; The type of distribution. 
            - Supported Types: 'linear', 'uniform', 'triangular', 'normal', and 'beta'.
        • interval: A list of two numbers specifying the low and high values on the x-axis to sample from.
        • coefficients: A list of two numbers. The meaning of the coefficients depends on the distribution type:
            - linear:     [slope, None]
            - sigmoid:    [slope, midpoint]
            - uniform:    [None,  None]
            - triangular: [mode,  None]
            - normal:     [mean,  variance]
            - beta:       [alpha, beta]
        • samples: list[float]; Array of floats randomly sampled values generated by sample_distribution()
        • export_fig: boolean; If True, the figure is saved to a file. Otherwise, it is displayed in the browser.

    Returns: 
        • void: Produces and potentially saves a Plotly figure, but returns nothing.
    """
    xval = np.linspace(interval[0], interval[1], 200)
    if distribution_type == "linear":
        yval = coefficients[0] * (xval - 0.5) + 0.5

    elif distribution_type == "sigmoid":
        yval = normal_sigmoid(xvals=np.linspace(interval[0], interval[1], 200), 
            slope=coefficients[0], midpoint=coefficients[1])
    
    elif distribution_type == "uniform":
        yval = np.ones(100) / (interval[1] - interval[0]) / 2
    
    elif distribution_type == "triangular":
        yval, angleC = np.zeros(100), (coefficients[0], 1)
        angleA, angleB = (interval[0], 0), (interval[1], 0)
        lineAC = np.where((xval >= angleA[0]) & (xval < angleC[0]))
        lineCB = np.where((xval >= angleC[0]) & (xval < angleB[0]))
        slopeAC = angleC[1] / (angleC[0] - angleA[0])
        interceptAC = angleA[1] - slopeAC * angleA[0]
        slopeCB = -angleC[1] / (angleB[0] - angleC[0])
        interceptCB = angleB[1] - slopeCB * angleB[0]
        yval[lineAC] = slopeAC * xval[lineAC] + interceptAC
        yval[lineCB] = slopeCB * xval[lineCB] + interceptCB
    
    elif distribution_type == "normal":
        yval = np.exp(-((xval - coefficients[0]) ** 2) / (2 * \
            coefficients[1])) / np.sqrt(2 * np.pi * coefficients[1])
    
    elif distribution_type == "beta":
        alpha, beta = coefficients[0], coefficients[1]
        yval = xval ** (alpha - 1) * (1 - xval) ** (beta - 1) / \
            (np.power(xval, alpha - 1) + np.power(1 - xval, beta - 1))
        yval /= np.trapz(yval, xval)
        yval = normalize(array_=yval)
    
    else: raise ValueError("Distribution type not supported")

    distribution_type = distribution_type.capitalize()
    rand_color = random.randrange(360)

    ideal_distribution = go.Scatter(x=xval, y=yval, name=f"{distribution_type} Distribution", 
        line=dict(color=hsla(hue=rand_color, saturation=100, lightness=50, alpha=0.6), width=12), 
        hovertemplate=f"<b>{distribution_type}<br>Distribution</b><br><br>" + "x: %{x:.2f}<br>" + "y: %{y:.2f}<br>")
    sample_distribution = go.Histogram(x=samples, xbins=dict(start=interval[0], end=interval[1]), nbinsx=30, 
        name="Sample Distribution", marker=dict(color=hsla(hue=rand_color+35, saturation=100, lightness=50, 
        alpha=0.6)), hovertemplate=f"<b>Bin</b><br><br>" + "x: %{x:.2f}<br>" + "y: %{y}<br>")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(template="plotly_dark", font=dict(family="Calibri", size=22))  
    fig.add_trace(sample_distribution, secondary_y=False)
    fig.add_trace(ideal_distribution, secondary_y=True)

    fig.update_xaxes(range=interval, tickfont=dict(size=20), 
        title=dict(text="Probability", font=dict(size=28)))
    if interval == [0, 1] or interval == [0.0, 1.0]:
        fig.update_xaxes(range=[0, 1], tickvals=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    fig.update_yaxes(range=[0, None], tickfont=dict(size=20), secondary_y=False, 
        title=dict(text="Sample Distribution", font=dict(size=28)), showgrid=False)
    fig.update_yaxes(range=[0, 1],    tickfont=dict(size=20), secondary_y=True,  
        title=dict(text=f"{distribution_type} Distribution", font=dict(size=28)))
    fig.update_layout(title=f"Sample {distribution_type} Distribution", 
        title_x=0.5, title_y=0.98, titlefont_size=30, margin=dict(r=0), 
        hoverlabel=dict(font_size=30), legend=dict(orientation="h"))

    file_name = edit_file_name(file_name=f"Sample_{distribution_type}_Distribution_{interval}{coefficients}{len(samples)}")
    fig.write_html(os.path.join(file_path_outputs, file_name + ".html")) if export_fig else fig.show()


def prob_next_round(current_round_number: int, prob_slope: float = 0.5, prob_midpoint: int | float = 10, constant: float = 5.5452) -> float:
    """
    Determines the probability that the experiment will continue to the next round given the current round number.
    
    Arguments:
        • current_round_number: int; Current round of the experiment.
        • prob_slope: float; The slope of the probability distribution.
        • prob_midpoint: int | float; X-axis value where the distribution intersects 
            the y-axis value of 0.5, meaning there is a 50% chance of a next round.
        • constant: float (default 5.5452); Adjusts slope.  If the slope and midpoint 
            are both 0.5, this constant ensures that the area under the curve between 
            0.0 and 0.5 is exactly double the area under the curve between 0.5 and 1.0.

    Returns:
        • probability: float ⊂ [0.0, 1.0]; The probability of the experiment progressing.
    """
    if not isinstance(current_round_number, int) or current_round_number < 0: 
        raise ValueError("The round number must be a positive integer.")
    
    return float(normal_sigmoid(
        xvals=current_round_number, 
        slope=prob_slope, 
        midpoint=prob_midpoint, 
        constant=constant)
        )


def simulate_number_of_experimental_rounds(num_samples: int, prob_slope: float = 0.5, prob_midpoint: int | float = 10, 
    constant: float = 5.5452, show_histogram: bool = False, export_fig= True, summary_stats= False) -> list[float] | dict | None:
    """
    Given the slope, midpoint, and constant, return the distribution over number of rounds per experiment.
    """
    rounds_per_experiment = []
    for sample in range(num_samples):
        round_ = 0
        while True:
            round_ += 1
            if random.random() > prob_next_round(current_round_number=round_, 
                prob_slope=prob_slope, prob_midpoint=prob_midpoint, constant=constant):
                rounds_per_experiment.append(round_)
                break    

    if show_histogram and num_samples > 200:      
        fig = go.Figure()
        rand_color = random.randrange(360)
        fig.add_trace(go.Histogram(x=rounds_per_experiment, name=f"N-Rounds<br>Histogram",
            marker=dict(color=hsla(hue=rand_color, saturation=100, lightness=50, alpha=0.6)), 
            hovertemplate=f"<b>Bin</b><br><br>" + "rounds: %{x}<br>" + "count: %{y}<br>"))
        fig.update_xaxes(title='N rounds per experiment', tickfont=dict(size=20))
        fig.update_yaxes(title='Count of Experiments', tickfont=dict(size=20))
        fig.update_layout(title='Distribution of N Rounds per Experiment', title_x=0.5, title_y=0.98, 
            titlefont_size=30, hoverlabel=dict(font_size=30), legend=dict(orientation="h"), 
            template="plotly_dark", font=dict(family='Calibri', size=22))

        file_name = edit_file_name(file_name=f"Nrounds_Histogram_slope{prob_slope}_midpoint{prob_midpoint}_constant{constant}")
        fig.write_html(os.path.join(file_path_outputs, file_name + ".html")) if export_fig else fig.show()

    if summary_stats:   
        summary = {
            'mean':     np.mean(rounds_per_experiment), 
            'median':   np.median(rounds_per_experiment), 
            'variance': np.var(rounds_per_experiment), 
            'std_dev':  np.std(rounds_per_experiment), 
            'skewness': sp.stats.skew(rounds_per_experiment), 
            'kurtosis': sp.stats.kurtosis(rounds_per_experiment)}
        return {key: round(val, 4) for key, val in summary.items()}
    
    return rounds_per_experiment


"Comment these lines in to try this"
# distribution_type, interval, coefficients, size = distribution_types[1], [0, 1], [0.5, 0.5], 5000
# samples = sample_distribution(distribution_type=distribution_type, 
#     interval=interval, coefficients=coefficients, size=size)
# print(f"Samples: {samples}")
# plot_distribution(distribution_type=distribution_type, interval=interval, 
#     coefficients=coefficients, samples=samples, export_fig=True)
# normal_sigmoid(xvals=np.linspace(0, 1, 1000), slope=0.5, midpoint=0.5, constant=5.5452, show=True)
# rounds_per_experiment = simulate_number_of_experimental_rounds(num_samples=3000, 
#     prob_slope=0.1, prob_midpoint=40, constant=5.5452, show_histogram=True, summary_stats=True)
# print(f"Rounds per Experiment: {rounds_per_experiment}")