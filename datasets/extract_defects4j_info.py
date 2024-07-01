import os
import json
import subprocess
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_project_ids():
    cmd = ['defects4j', 'pids']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    project_ids = result.stdout.splitlines()
    return project_ids

def get_bug_ids(project_id):
    cmd = ['defects4j', 'bids', '-p', project_id]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    bug_ids = result.stdout.splitlines()
    return bug_ids

def checkout(project_id, bug_id):
    checkout_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id) + 'b')
    shutil.rmtree(checkout_path, ignore_errors=True)
    cmd = ['defects4j', 'checkout', '-p', project_id, '-v', str(bug_id) + 'b', '-w', checkout_path]
    print(" ".join(cmd))
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    checkout_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id) + 'f')
    shutil.rmtree(checkout_path, ignore_errors=True)
    cmd = ['defects4j', 'checkout', '-p', project_id, '-v', str(bug_id) + 'f', '-w', checkout_path]
    print(" ".join(cmd))
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def query_info(project_id):
    save_path = os.path.join(FILE_DIR, "projects", project_id, "query_info.csv")
    query_str = "bug.id,revision.id.buggy,revision.id.fixed,report.id,report.url,deprecated.version,deprecated.reason,classes.relevant.src,classes.relevant.test,classes.modified,tests.relevant,tests.trigger,tests.trigger.cause,project.id,project.name,project.build.file,project.vcs,project.repository,project.bugs.csv,revision.date.buggy,revision.date.fixed"
    cmd = ['defects4j', 'query', '-p', project_id, '-q', query_str, '-o', save_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["sed", "-i", "1i"+query_str, save_path])
    
def export_info(project_id, bug_id):
    checkout_path = os.path.join(FILE_DIR, "projects", project_id, str(bug_id) + 'b')
    cmd = ['defects4j', 'export', '-p', 'tests.trigger', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tests_trigger = result.stdout.splitlines()
    cmd = ['defects4j', 'export', '-p', 'dir.src.classes', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    src_classes = result.stdout.splitlines()[-1].strip()
    cmd = ['defects4j', 'export', '-p', 'dir.src.tests', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    src_tests = result.stdout.splitlines()[-1].strip()
    cmd = ['defects4j', 'export', '-p', 'tests.relevant', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tests_relevant = result.stdout.splitlines()
    cmd = ['defects4j', 'export', '-p', 'tests.all', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tests_all = result.stdout.splitlines()
    cmd = ['defects4j', 'export', '-p', 'classes.modified', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    classes_modified = result.stdout.splitlines()
    cmd = ['defects4j', 'export', '-p', 'classes.relevant', '-w', checkout_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    classes_relevant = result.stdout.splitlines()
    
    return {
        "project_id": project_id,
        "bug_id": bug_id,
        "src_classes": src_classes,
        "src_tests": src_tests,
        "tests_trigger": tests_trigger,
        "tests_relevant": tests_relevant,
        "tests_all": tests_all,
        "classes_modified": classes_modified,
        "classes_relevant": classes_relevant,
    }
    
if __name__ == "__main__":
    project_bug_ids, task_info = [], []
    
    project_ids = get_project_ids()
    for project_id in project_ids:
        bug_ids = get_bug_ids(project_id)
        # print(bug_ids)
        project_bug_ids.extend([(project_id, bug_id) for bug_id in bug_ids])    
    
    with ProcessPoolExecutor(max_workers=max(os.cpu_count()//4, 1)) as executor:
        results = list(tqdm(executor.map(checkout, *zip(*project_bug_ids)), total=len(project_bug_ids), desc="Checkout"))
        task_info = list(tqdm(executor.map(export_info, *zip(*project_bug_ids)), total=len(project_bug_ids), desc="Export"))
    
    for project_id in project_ids:
        query_info(project_id)

    if task_info:
        os.makedirs(os.path.join(FILE_DIR, "data"), exist_ok=True)
        json.dump(task_info, open(os.path.join(FILE_DIR, "data", "task_info.json"), "w"), indent=2)
    
    task_info = json.load(open(os.path.join(FILE_DIR, "data", "task_info.json"), "r"))
    
    for task in task_info:
        print(task["project_id"], task["bug_id"])
        print("src_classes", task["src_classes"])
        print("src_tests", task["src_tests"])
        print("tests_relevant", task["tests_relevant"])
        print("classes_modified", task["classes_modified"])
        break
    
    print("Done")
    