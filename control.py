import os
import csv
import time
import glob
from gurobipy import GRB
from parser import parse_json_from_file
from MILP import create_model  # passe ggf. den Modulnamen an


############ CONFIGURATION ############
alpha_value = 0.1
pre_filter = False
pre_filter_tolerance = 0.6
time_limit = 3600
threads = 1
#######################################


def solve_instance(name, log_path):
    (pages, article, layouts, sections, article_sections, sections_page,
     layouts_pages, box_layouts, box_geomtery, shells_layout_box, shells_article,
     article_length, shell_params, article_priority) = parse_json_from_file(name)

    model, shells_layout_box_article, x, y = create_model(
        pages, article, layouts, sections, article_sections, sections_page,
        layouts_pages, box_layouts, shells_layout_box, shells_article, article_length,
        shell_params, article_priority, alpha_value, pre_filter, pre_filter_tolerance
    )

    model.setParam('TimeLimit', time_limit)
    model.Params.Threads = threads
    model.Params.OutputFlag = 1  # weniger Konsolen-Spam; auf 1 setzen falls gewünscht
    model.Params.LogFile = log_path

    model.optimize()
    return model


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_dir = os.path.join(base_dir, "Instances", "realdeal")
    sol_dir = os.path.join(instance_dir, "sol")
    log_dir = os.path.join(instance_dir, "log")
    os.makedirs(sol_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    summary_path = os.path.join(instance_dir, "summary.csv")

    instance_files = sorted(glob.glob(os.path.join(instance_dir, "*.json")))
    print(f"{len(instance_files)} Instanzen gefunden.\n")

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance", "status", "objective", "best_bound",
            "gap_percent", "runtime_seconds", "num_vars", "num_constrs"
        ])

        for idx, name in enumerate(instance_files, 1):
            instance_name = os.path.splitext(os.path.basename(name))[0]
            print(f"[{idx}/{len(instance_files)}] {instance_name} ...", flush=True)

            log_path = os.path.join(log_dir, f"{instance_name}.log")
            start = time.time()
            model = None
            try:
                model = solve_instance(name, log_path)
                runtime = time.time() - start

                status_map = {
                    GRB.OPTIMAL: "OPTIMAL",
                    GRB.TIME_LIMIT: "TIME_LIMIT",
                    GRB.INFEASIBLE: "INFEASIBLE",
                    GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    GRB.UNBOUNDED: "UNBOUNDED",
                }
                status = status_map.get(model.status, str(model.status))

                if model.SolCount > 0:
                    obj = model.ObjVal
                    bound = model.ObjBound
                    gap = model.MIPGap * 100
                    model.write(os.path.join(sol_dir, f"{instance_name}.sol"))
                else:
                    obj = ""
                    bound = model.ObjBound if model.status != GRB.INFEASIBLE else ""
                    gap = ""

                writer.writerow([
                    instance_name, status,
                    f"{obj:.6f}" if obj != "" else "",
                    f"{bound:.6f}" if bound != "" else "",
                    f"{gap:.4f}" if gap != "" else "",
                    f"{runtime:.2f}",
                    model.NumVars, model.NumConstrs
                ])
                f.flush()
                print(f"    {status} | obj={obj} | gap={gap} | {runtime:.1f}s", flush=True)

            except Exception as e:
                runtime = time.time() - start
                writer.writerow([instance_name, f"ERROR: {e}", "", "", "", f"{runtime:.2f}", "", ""])
                f.flush()
                print(f"    FEHLER: {e}", flush=True)

            finally:
                # Modelle akkumulieren sonst über den Lauf hinweg Speicher (bei 18
                # grossen Instanzen hintereinander führte das zu spürbarer
                # Verlangsamung gegen Ende des Batches) -> explizit freigeben.
                if model is not None:
                    model.dispose()

    print(f"\nFertig. Übersicht: {summary_path}")

