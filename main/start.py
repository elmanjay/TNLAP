from parser import parse_json_from_file
import os
#from analyse import analyse_solution
#from analyse_ressorts import analyse_solution
#from analyse_bugs import check_constraint_violations
from tnlap import create_model
#from tnlap_collapsed import create_model
#from tnlap_dynamic import create_model
from gurobipy import GRB
#from analyse_new import analyse_solution
import pandas as pd


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(base_dir, "Instances")
    instance_instance_dir = os.path.join(instance_dir, "Test")

    lp_dir = os.path.join(instance_dir, "lp")
    lp_mps_dir = os.path.join(lp_dir, "mps")
    sol_dir = os.path.join(instance_dir, "sol")

    os.makedirs(lp_dir, exist_ok=True)
    os.makedirs(lp_mps_dir, exist_ok=True)
    os.makedirs(sol_dir, exist_ok=True)

    alpha_value = 0.4 
    pre_f = False
    pre_f_tol = 0.3

    # 🔁 Instanzen definieren
    instance_base = "TestP1A3V{}(B)"
    instances = [instance_base.format(i) for i in range(1, 2)]#6

    results = []

    for instance_name in instances:
        print(f"Starte Instanz: {instance_name}")

        name = os.path.join(instance_instance_dir, f"{instance_name}.json")

        pages, article, layouts, sections, article_sections, sections_page, layouts_pages, box_layouts, geometry_layout_box, shells_layout_box, shells_article, article_length, shell_params, article_priority, layout_vertical_chains = parse_json_from_file(name)
        
        print("start building model...")
        model = create_model(
            pages, article, layouts, sections, article_sections, sections_page, layouts_pages,
            box_layouts, shells_layout_box, shells_article, article_length, shell_params,
            article_priority, alpha_value, pre_f, pre_f_tol, geometry_layout_box, layout_vertical_chains #NEU
        )
      
        print("model completed. start optimizing...")
        model.setParam('TimeLimit', 3600)
        model.Params.LogFile = os.path.join(sol_dir, f"{instance_name}.log")
        model.Params.Threads = 1
        #model.Params.NonConvex = 2 #### aktivieren für nicht lineares dynamisches Modell
        model.write(os.path.join(lp_dir, f"{instance_name}.lp"))
        #model.Params.OutputFlag = 0  # 🔇 KEIN TERMINAL-OUTPUT
        model.optimize()

        # 📊 Ergebnisse speichern
        if model.status == GRB.INFEASIBLE:
            print("Modell ist unloesbar.")
            obj = None
            runtime = model.Runtime
            status = "INFEASIBLE"
        else:
            print("Modell gelöst.")
            obj = model.ObjVal
            runtime = model.Runtime
            status = "OPTIMAL" if model.status == GRB.OPTIMAL else "TIME_LIMIT"

            model.write(os.path.join(lp_dir, f"{instance_name}.lp"))
            model.write(os.path.join(sol_dir, f"{instance_name}.sol"))
            #analyse_solution(model, article_length, hull_params, article_priority)
            #analyse_solution(model, pages, layouts_pages, box_layouts, geometry_layout_box, layout_vertical_chains, hull_params)
            #y_values = {}
            #for v in model.getVars():
                #if v.VarName.startswith("y_") and abs(v.X - 1) <= 1e-9:
                    #_, j, l = v.VarName.split("_")
                    #y_values[(int(j), int(l))] = v.X
            #check_constraint_violations(model=model, y_values=y_values)

        results.append({
            "Instance": instance_name,
            "Runtime (s)": runtime,
            "Objective": obj,
            "Status": status
        })

    
    # 📁 Excel schreiben
    df = pd.DataFrame(results)
    output_path = os.path.join(instance_dir, "results.xlsx")
    df.to_excel(output_path, index=False)

    print(f"Ergebnisse gespeichert in: {output_path}")
