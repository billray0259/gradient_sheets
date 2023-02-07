import dash
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
from gradient_sheets import get_tensor_sheet, get_grads, percent_grads
from sheet_graph import find_edges, get_labels, get_node_locations_v2, count_crossings

import gspread


gc = gspread.oauth()
app = dash.Dash(external_stylesheets=["https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/spacelab/bootstrap.min.css"])


sh = gc.open_by_key("1dGNB0YbWVPcD5-GDD9WDaFj9K3o0qppFLEuOK6e8CoI")
worksheet = sh.sheet1

formula_sheet = worksheet.get_values(value_render_option='FORMULA')

edges = find_edges(formula_sheet)

node_locations = get_node_locations_v2(edges)

cell_names = list(set([edge[0] for edge in edges] + [edge[1] for edge in edges]))

labels = get_labels(cell_names, formula_sheet)
elements = [
    {
        'data': {
            'id': cell_name,
            'label': label
        }
    } 
    for cell_name, label in zip(cell_names, labels)
]

elements += [
    {
        'data': {
            'id': f'{edge[0]}-{edge[1]}',
            'source': edge[0], 
            'target': edge[1]
        }
    }
    for edge in edges
]

default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            "text-wrap": "wrap",
            "text-max-width": 80,
            "background-color": "#A0A28F",
            "color": "white",
            "font-family": "monospace",
            "text-background-opacity": 0.7,
            "text-background-color": "#272822",
            # make text smaller
            "font-size": 10,
        }
    },
    {
        "selector": "edge",
        "style": {
            "mid-target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "line-color": "#888470",
            "target-arrow-color": "#888470"
        }
    }
]

app.layout = cyto.Cytoscape(
    id="cytoscape",

    # layout={"name": "cose"}, use a preset layout instead of cose
    layout={
        "name": "preset",
        "positions": {
            cell_name: {
                'x': node_locations[cell_name][0]*100,
                'y': node_locations[cell_name][1]*75
            }
            for cell_name in cell_names
        }    
    },
    stylesheet=default_stylesheet,
    style={
        'width': '100%',
        'height': '100vh',
        'border-style':
        'solid',
        "background-color": "#272822"
    },
    elements=elements
)


def get_color(grad, min_grad, max_grad):
    # a big positive gradient is bright green
    # a big negative gradient is bright red
    # a small positive gradient is dark green
    # a small negative gradient is dark red
    # colors are continuous
    # 0 is gray


    if grad > 0:
        grad = (grad - 0) / (max_grad - 0)
        return (0, 235 * grad + 20, 0)
    elif grad < 0:
        grad = -grad
        grad = (grad - 0) / ((-min_grad) - 0)
        return (235 * grad + 20, 0, 0)
    else:
        color = (160, 162, 143)
    
    return color



@app.callback(
    Output("cytoscape", "stylesheet"),
    Input("cytoscape", "tapNodeData"),
)
def update_stylesheet(node_data):
    if node_data is None:
        return default_stylesheet

    tensor_sheet = get_tensor_sheet(worksheet.get_values(value_render_option='FORMULA'))
    clicked_cell_name = node_data['id']
    grads = get_grads(tensor_sheet, clicked_cell_name, leaf_only=False)
    p_grads = percent_grads(grads, tensor_sheet, node_data['id'])

    stylesheet = default_stylesheet.copy()
    cell_names = list(p_grads.keys())
    grads = list(p_grads.values())
    labels = get_labels(cell_names, formula_sheet)

    for cell_name, grad, label in zip(cell_names, grads, labels):
        if cell_name == clicked_cell_name:
            continue

        color = get_color(grad, min(p_grads.values()), max(p_grads.values()))

        stylesheet.append({
            'selector': f'node[id = "{cell_name}"]',
            'style': {
                'background-color': color,
                'content': f'{grad:.2f}\n{label}'
            }
        })
    
    # make clicked node gold
    stylesheet.append({
        'selector': f'node[id = "{clicked_cell_name}"]',
        'style': {
            'background-color': (255, 215, 0)
        }
    })

    return stylesheet



if __name__ == "__main__":
    app.run_server(debug=True, port=8000)