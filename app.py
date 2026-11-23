"""
Deep Q-Network — Aprendizado por Reforço
Dashboard Flask com duas abas: Computação Clássica e Computação Quântica.

Autor: Pedro Paulo Ribeiro — www.inforpop.com
"""
import os

from flask import Flask, jsonify, render_template, request

from rl.quantum import VQC
from rl.trainer import DEFAULTS, JobManager

app = Flask(__name__)
manager = JobManager()

CONTATOS = {
    "site": "https://www.inforpop.com",
    "site_label": "www.inforpop.com",
    "whatsapp": "https://wa.me/5521999618643",
    "whatsapp_label": "55 21 99961-8643",
    "email": "inforpop@hotmail.com",
    "linkedin": "https://www.linkedin.com/in/pedro-paulo-ribeiro-pcd-1451498/",
    "linkedin_label": "Pedro Paulo Ribeiro",
}


@app.route("/")
def index():
    return render_template("index.html", c=CONTATOS)


@app.route("/health")
def health():
    return jsonify(ok=True, running=manager.running())


@app.route("/api/defaults")
def defaults():
    return jsonify(DEFAULTS)


@app.route("/api/train", methods=["POST"])
def train():
    data = request.get_json(silent=True) or {}
    try:
        job = manager.start(data.get("mode"), data.get("env"),
                            data.get("config", {}), data.get("seed"))
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(job_id=job.id, config=job.cfg, seed=job.seed)


def _job_or_404(job_id):
    job = manager.get(job_id)
    if job is None:
        return None, (jsonify(error="treino não encontrado (o servidor pode ter reiniciado)"), 404)
    return job, None


@app.route("/api/status/<job_id>")
def status(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    since = request.args.get("since", default=0, type=int)
    return jsonify(job.snapshot(max(since, 0)))


@app.route("/api/stop/<job_id>", methods=["POST"])
def stop(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    job.stop_flag.set()
    return jsonify(ok=True)


@app.route("/api/policy/<job_id>")
def policy(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    which = request.args.get("model", "best")
    return jsonify(job.policy("last" if which == "last" else "best"))


@app.route("/api/statevector/<job_id>")
def statevector(job_id):
    job, err = _job_or_404(job_id)
    if err:
        return err
    return jsonify(job.statevector(request.args.get("s", default=0, type=int)))


@app.route("/api/circuit")
def circuit():
    layers = min(max(request.args.get("layers", default=3, type=int), 1), 8)
    env = request.args.get("env", "gridworld")
    enc = "angle" if env == "cartpole" else "basis"
    return jsonify(VQC(4, layers, enc).describe())


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
