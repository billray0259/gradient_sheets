from dash import Dash, html, dcc, Input, Output, State
import dash_cytoscape as cyto
from gradient_sheets import get_tensor_sheet, get_grads, percent_grads
from sheet_graph import build_cytoscape_component

import dash_bootstrap_components as dbc

import gspread


spreadsheet_key = "1dGNB0YbWVPcD5-GDD9WDaFj9K3o0qppFLEuOK6e8CoI"

# gc = gspread.oauth()
app = Dash("Gradient Sheets", external_stylesheets=["https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/spacelab/bootstrap.min.css"])


app.layout = html.Div([
    dcc.Store(id="formula_sheet"),
    html.Div(
        [
            dbc.Row(
                [   
                    # title
                    dbc.Col(
                        [
                            html.H1("Gradient Sheets")
                        ],
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    # border red for debugging
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [
                    # File, Edit, View, Insert, Format, Data, Tools, Add-ons, Help dropdowns
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("New"),
                            dbc.DropdownMenuItem("Open..."),
                            dbc.DropdownMenuItem("Save"),
                            dbc.DropdownMenuItem("Save As..."),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Print..."),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Exit"),
                        ],
                        label="File",
                        id="file-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Undo"),
                            dbc.DropdownMenuItem("Redo"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Cut"),
                            dbc.DropdownMenuItem("Copy"),
                            dbc.DropdownMenuItem("Paste"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Find..."),
                            dbc.DropdownMenuItem("Find and Replace..."),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Select All"),
                        ],
                        label="Edit",
                        id="edit-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Zoom In"),
                            dbc.DropdownMenuItem("Zoom Out"),
                            dbc.DropdownMenuItem("Zoom to 100%"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Gridlines"),
                            dbc.DropdownMenuItem("Hide Gridlines"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Formulas"),
                            dbc.DropdownMenuItem("Hide Formulas"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Headers"),
                            dbc.DropdownMenuItem("Hide Headers"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Row Numbers"),
                            dbc.DropdownMenuItem("Hide Row Numbers"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Column Letters"),
                            dbc.DropdownMenuItem("Hide Column Letters"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Zero Values"),
                            dbc.DropdownMenuItem("Hide Zero Values"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Show Gridlines"),
                            dbc.DropdownMenuItem("Hide Gridlines"),
                        ],
                        label="View",
                        id="view-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Insert"),
                            dbc.DropdownMenuItem("Delete"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Insert Cells..."),
                            dbc.DropdownMenuItem("Delete Cells..."),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Insert Rows"),
                            dbc.DropdownMenuItem("Insert Columns"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Delete Rows"),
                            dbc.DropdownMenuItem("Delete Columns"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Insert Sheet"),
                            dbc.DropdownMenuItem("Delete Sheet"),
                        ],
                        label="Insert",
                        id="insert-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Font"),
                            dbc.DropdownMenuItem("Font Color"),
                            dbc.DropdownMenuItem("Fill Color"),
                            dbc.DropdownMenuItem("Borders"),
                            dbc.DropdownMenuItem("Alignment"),
                            dbc.DropdownMenuItem("Number Format"),
                            dbc.DropdownMenuItem("Text Format"),
                            dbc.DropdownMenuItem("Merge Cells"),
                            dbc.DropdownMenuItem("Unmerge Cells"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Insert Function"),
                        ],
                        label="Format",
                        id="format-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                        [
                            dbc.DropdownMenuItem("Sort"),
                            dbc.DropdownMenuItem("Filter"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Group"),
                            dbc.DropdownMenuItem("Ungroup"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Clear"),
                            dbc.DropdownMenuItem("Clear Contents"),
                            dbc.DropdownMenuItem("Clear Formats"),
                            dbc.DropdownMenuItem("Clear Comments"),
                            dbc.DropdownMenuItem("Clear Hyperlinks"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Clear All"),
                        ],
                        label="Data",
                        id="data-dropdown",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dbc.DropdownMenu(
                    [
                        dbc.DropdownMenuItem("Show/Hide"),
                        dbc.DropdownMenuItem("Rename"),
                        dbc.DropdownMenuItem("Move or Copy"),
                        dbc.DropdownMenuItem("Insert"),
                        dbc.DropdownMenuItem("Delete"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Protect Sheet"),
                        dbc.DropdownMenuItem("Unprotect Sheet"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem("Protect Workbook"),
                        dbc.DropdownMenuItem("Unprotect Workbook"),
                    ],
                    label="Sheet",
                    id="sheet-dropdown",
                    style={
                            "display": "inline-block",
                        },
                    ),
                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [

                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [

                ],
                style={
                    "width": "100%",
                    "height": "5hv",
                    "border": "1px solid red"
                },
            ),
            dbc.Row(
                [
                    "hello"
                ],
                style={
                    "width": "100%",
                    "height": "80hv",
                    "border": "1px solid red"
                },
            ),
        ],
        style={
            "width": "100%",
            "height": "100hv"
        }
    ),
])



    



# @app.callback(
#     Output("cytoscape", "stylesheet"),
#     Input("cytoscape", "tapNodeData"),
#     State("cytoscape", "stylesheet")
# )
# def update_stylesheet(node_data, default_stylesheet):
#     if node_data is None:
#         return default_stylesheet

#     tensor_sheet = get_tensor_sheet(worksheet.get_values(value_render_option='FORMULA'))
#     clicked_cell_name = node_data['id']
#     grads = get_grads(tensor_sheet, clicked_cell_name, leaf_only=False)
#     p_grads = percent_grads(grads, tensor_sheet, node_data['id'])

#     stylesheet = default_stylesheet
#     cell_names = list(p_grads.keys())
#     grads = list(p_grads.values())
#     labels = get_labels(cell_names, formula_sheet)

#     for cell_name, grad, label in zip(cell_names, grads, labels):
#         if cell_name == clicked_cell_name:
#             continue

#         max_grad = max(p_grads.values())
#         min_grad = min(p_grads.values())

#         if grad > 0:
#             grad = (grad - 0) / (max_grad - 0)
#             return (0, 235 * grad + 20, 0)
#         elif grad < 0:
#             grad = -grad
#             grad = (grad - 0) / ((-min_grad) - 0)
#             return (235 * grad + 20, 0, 0)
#         else:
#             color = (160, 162, 143)

#         stylesheet.append({
#             'selector': f'node[id = "{cell_name}"]',
#             'style': {
#                 'background-color': color,
#                 'content': f'{grad:.2f}\n{label}'
#             }
#         })
    
#     # make clicked node gold
#     stylesheet.append({
#         'selector': f'node[id = "{clicked_cell_name}"]',
#         'style': {
#             'background-color': (255, 215, 0)
#         }
#     })

#     return stylesheet



if __name__ == "__main__":
    app.run_server(debug=True, port=8000)