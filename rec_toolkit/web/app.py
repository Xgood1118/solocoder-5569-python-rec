import os
import uuid
import threading
import time
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS

from ..recommender import RecommenderSystem
from ..data import DataLoader
from ..config import config
from ..scheduler.scheduler import Scheduler, IncrementalUpdater


training_status = {
    'status': 'idle',
    'message': '',
    'progress': 0,
    'job_id': None,
    'started_at': None,
    'finished_at': None,
}


def _run_training_job(job_id: str, task_func, task_name: str, rec_system):
    global training_status
    try:
        training_status['status'] = 'training'
        training_status['message'] = f'{task_name}中...'
        training_status['progress'] = 10
        training_status['job_id'] = job_id
        training_status['started_at'] = time.time()
        training_status['finished_at'] = None

        task_func()

        training_status['status'] = 'done'
        training_status['message'] = f'{task_name}完成'
        training_status['progress'] = 100
        training_status['finished_at'] = time.time()
    except Exception as e:
        training_status['status'] = 'error'
        training_status['message'] = f'{task_name}失败: {str(e)}'
        training_status['finished_at'] = time.time()


def create_app(data_dir: str = None) -> Flask:
    if data_dir is None:
        data_dir = config['data']['data_dir']

    app = Flask(__name__, static_folder=None)
    CORS(app)

    os.makedirs(data_dir, exist_ok=True)
    rec_system = RecommenderSystem(data_dir=data_dir)

    scheduler = Scheduler()
    incremental_updater = IncrementalUpdater(
        rec_system,
        interval_seconds=config['scheduler']['incremental_interval']
    )

    scheduler.add_job(
        'incremental_update',
        incremental_updater.update,
        interval_seconds=config['scheduler']['incremental_interval']
    )
    scheduler.add_job(
        'retrain',
        incremental_updater.retrain,
        cron_expr=config['scheduler']['retrain_cron']
    )
    scheduler.start()

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>推荐系统演示 - rec_toolkit</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #f5f7fa; color: #333; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 30px; text-align: center; }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .card { background: white; border-radius: 12px; padding: 24px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .card h2 { font-size: 18px; margin-bottom: 16px; color: #333;
                   border-bottom: 2px solid #667eea; padding-bottom: 8px; display: inline-block; }
        .row { display: flex; gap: 16px; flex-wrap: wrap; }
        .col { flex: 1; min-width: 250px; }
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; margin-bottom: 6px; font-weight: 500; font-size: 14px; }
        .form-group input, .form-group select, .form-group textarea {
            width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px;
            font-size: 14px; outline: none; transition: border-color 0.2s;
        }
        .form-group input:focus, .form-group select:focus { border-color: #667eea; }
        .btn { background: #667eea; color: white; border: none; padding: 10px 20px;
               border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500;
               transition: background 0.2s; }
        .btn:hover:not(:disabled) { background: #5a67d8; }
        .btn:disabled { background: #a0aec0; cursor: not-allowed; opacity: 0.7; }
        .btn-secondary { background: #718096; }
        .btn-secondary:hover:not(:disabled) { background: #4a5568; }
        .btn-small { padding: 6px 12px; font-size: 12px; }
        .btn-danger { background: #e53e3e; }
        .btn-danger:hover:not(:disabled) { background: #c53030; }
        .recommendation-list { list-style: none; }
        .recommendation-item { display: flex; align-items: center; padding: 12px;
                                border-radius: 8px; margin-bottom: 8px; background: #f7fafc;
                                transition: background 0.2s; }
        .recommendation-item:hover { background: #edf2f7; }
        .item-rank { width: 30px; height: 30px; border-radius: 50%;
                     background: #667eea; color: white; display: flex;
                     align-items: center; justify-content: center; font-weight: bold;
                     font-size: 14px; margin-right: 12px; }
        .item-info { flex: 1; }
        .item-id { font-weight: 500; font-size: 14px; color: #2d3748; }
        .item-meta { font-size: 12px; color: #718096; margin-top: 4px; }
        .item-reason { font-size: 12px; color: #667eea; margin-top: 4px;
                       background: #ebf4ff; padding: 2px 8px; border-radius: 4px;
                       display: inline-block; }
        .item-score { font-size: 12px; color: #48bb78; font-weight: 500; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                      gap: 16px; }
        .stat-item { background: #f7fafc; padding: 16px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 12px; color: #718096; margin-top: 4px; }
        .tab-bar { display: flex; gap: 4px; margin-bottom: 16px;
                   border-bottom: 2px solid #e2e8f0; padding-bottom: 0; flex-wrap: wrap; }
        .tab-item { padding: 10px 20px; cursor: pointer; font-size: 14px;
                    color: #718096; border-bottom: 2px solid transparent;
                    margin-bottom: -2px; transition: all 0.2s; }
        .tab-item.active { color: #667eea; border-bottom-color: #667eea; font-weight: 500; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .channel-weight { display: flex; align-items: center; margin-bottom: 10px; }
        .channel-weight label { width: 100px; font-size: 13px; }
        .channel-weight input[type=range] { flex: 1; margin: 0 10px; }
        .channel-weight span { width: 40px; text-align: right; font-size: 13px; color: #667eea; }
        .alert { padding: 12px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
        .alert-success { background: #f0fff4; color: #22543d; border: 1px solid #9ae6b4; }
        .alert-error { background: #fff5f5; color: #742a2a; border: 1px solid #feb2b2; }
        .alert-info { background: #ebf8ff; color: #2a4365; border: 1px solid #90cdf4; }
        .evaluation-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .evaluation-table th, .evaluation-table td {
            padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0;
        }
        .evaluation-table th { background: #f7fafc; font-weight: 500; color: #4a5568; }
        .upload-area { border: 2px dashed #cbd5e0; border-radius: 8px;
                       padding: 30px; text-align: center; color: #718096;
                       cursor: pointer; transition: border-color 0.2s; }
        .upload-area:hover { border-color: #667eea; }
        .upload-area.dragover { border-color: #667eea; background: #f7fafc; }
        .config-display { background: #f7fafc; padding: 16px; border-radius: 8px; }
        .config-display .config-row { display: flex; padding: 8px 0; border-bottom: 1px solid #e2e8f0; }
        .config-display .config-row:last-child { border-bottom: none; }
        .config-display .config-key { width: 120px; font-weight: 500; color: #4a5568; font-size: 14px; }
        .config-display .config-value { flex: 1; color: #2d3748; font-size: 14px; font-family: monospace; }
        .scheduler-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .scheduler-table th, .scheduler-table td {
            padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0;
        }
        .scheduler-table th { background: #f7fafc; font-weight: 500; color: #4a5568; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 500; }
        .badge-running { background: #c6f6d5; color: #22543d; }
        .badge-stopped { background: #fed7d7; color: #742a2a; }
        .badge-idle { background: #e2e8f0; color: #2d3748; }
        .badge-training { background: #fefcbf; color: #744210; }
        .badge-done { background: #c6f6d5; color: #22543d; }
        .badge-error { background: #fed7d7; color: #742a2a; }
        .progress-bar-container { width: 100%; height: 6px; background: #e2e8f0;
                                   border-radius: 3px; overflow: hidden; margin-top: 8px; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2);
                        border-radius: 3px; transition: width 0.3s; }
        .spinner { display: inline-block; width: 14px; height: 14px;
                   border: 2px solid rgba(255,255,255,0.3); border-top-color: white;
                   border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 6px;
                   vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .training-status-box { background: #f7fafc; padding: 12px 16px; border-radius: 8px;
                               margin-bottom: 16px; display: none; }
        .training-status-box.visible { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 推荐系统演示平台</h1>
        <p>轻量级推荐系统 - 支持多算法、多路召回、冷启动、Bandit优化</p>
    </div>

    <div class="container">
        <div class="card">
            <h2>系统概览</h2>
            <div id="training-status" class="training-status-box">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span id="ts-badge" class="badge badge-idle">空闲</span>
                        <span id="ts-message" style="font-size: 14px;">暂无训练任务</span>
                    </div>
                </div>
                <div class="progress-bar-container">
                    <div id="ts-progress" class="progress-bar" style="width: 0%"></div>
                </div>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value" id="stat-users">-</div>
                    <div class="stat-label">用户数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="stat-items">-</div>
                    <div class="stat-label">物品数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="stat-interactions">-</div>
                    <div class="stat-label">交互数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="stat-trained">-</div>
                    <div class="stat-label">训练状态</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="tab-bar">
                <div class="tab-item active" data-tab="recommend">个性化推荐</div>
                <div class="tab-item" data-tab="similar">相似物品</div>
                <div class="tab-item" data-tab="data">数据管理</div>
                <div class="tab-item" data-tab="config">参数配置</div>
                <div class="tab-item" data-tab="scheduler">系统调度</div>
                <div class="tab-item" data-tab="evaluate">评估指标</div>
            </div>

            <div class="tab-content active" id="tab-recommend">
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>用户ID</label>
                            <input type="text" id="user-id" value="user_1" placeholder="输入用户ID">
                        </div>
                        <div class="form-group">
                            <label>推荐数量</label>
                            <select id="rec-count">
                                <option value="5">5</option>
                                <option value="10" selected>10</option>
                                <option value="20">20</option>
                                <option value="50">50</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>上下文（可选）</label>
                            <select id="context-device">
                                <option value="">无</option>
                                <option value="mobile">移动端</option>
                                <option value="desktop">桌面端</option>
                                <option value="tablet">平板</option>
                            </select>
                        </div>
                        <button class="btn" onclick="getRecommendations()">获取推荐</button>
                        <button class="btn btn-secondary" onclick="simulateClick()">模拟点击反馈</button>
                    </div>
                    <div class="col">
                        <h3 style="font-size:16px; margin-bottom:12px;">推荐结果</h3>
                        <ul class="recommendation-list" id="rec-list">
                            <li style="color:#718096; text-align:center; padding:20px;">
                                点击"获取推荐"查看结果
                            </li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="tab-content" id="tab-similar">
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>物品ID</label>
                            <input type="text" id="item-id" value="item_1" placeholder="输入物品ID">
                        </div>
                        <div class="form-group">
                            <label>相似数量</label>
                            <select id="similar-count">
                                <option value="5" selected>5</option>
                                <option value="10">10</option>
                                <option value="20">20</option>
                            </select>
                        </div>
                        <button class="btn" onclick="getSimilarItems()">查找相似物品</button>
                    </div>
                    <div class="col">
                        <h3 style="font-size:16px; margin-bottom:12px;">相似物品</h3>
                        <ul class="recommendation-list" id="similar-list">
                            <li style="color:#718096; text-align:center; padding:20px;">
                                点击"查找相似物品"查看结果
                            </li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="tab-content" id="tab-data">
                <div class="row">
                    <div class="col">
                        <h3 style="font-size:16px; margin-bottom:12px;">CSV数据上传</h3>
                        <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                            <p>📁 点击上传或拖拽CSV文件</p>
                            <p style="font-size:12px; margin-top:8px;">支持 users.csv, items.csv, interactions.csv</p>
                        </div>
                        <input type="file" id="file-input" style="display:none;" accept=".csv" onchange="handleFileUpload(event)">
                        <div class="form-group" style="margin-top:16px;">
                            <label>数据类型</label>
                            <select id="data-type">
                                <option value="users">用户数据</option>
                                <option value="items">物品数据</option>
                                <option value="interactions">交互数据</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>导入模式</label>
                            <select id="import-mode">
                                <option value="batch">全量覆盖</option>
                                <option value="incremental">增量追加</option>
                            </select>
                        </div>
                        <div id="upload-result"></div>
                    </div>
                    <div class="col">
                        <h3 style="font-size:16px; margin-bottom:12px;">数据操作</h3>
                        <button class="btn" id="btn-sample" onclick="loadSampleData()" style="margin-right:8px;">加载示例数据</button>
                        <button class="btn btn-secondary" id="btn-retrain" onclick="retrainModel()">重新训练模型</button>
                        <div id="action-result" style="margin-top:16px;"></div>
                    </div>
                </div>
            </div>

            <div class="tab-content" id="tab-config">
                <h3 style="font-size:16px; margin-bottom:16px;">服务器配置</h3>
                <div class="config-display" id="server-config-display" style="margin-bottom:24px;"></div>

                <h3 style="font-size:16px; margin-bottom:16px;">召回通道权重配置</h3>
                <div id="channel-weights"></div>
                <button class="btn" onclick="saveChannelWeights()">保存配置</button>
                <div id="config-result" style="margin-top:16px;"></div>
            </div>

            <div class="tab-content" id="tab-scheduler">
                <h3 style="font-size:16px; margin-bottom:16px;">调度器控制</h3>
                <div class="row" style="margin-bottom:16px;">
                    <div class="col" style="flex: none;">
                        <button class="btn" id="btn-scheduler-start" onclick="startScheduler()">启动调度器</button>
                        <button class="btn btn-danger" id="btn-scheduler-stop" onclick="stopScheduler()" style="margin-left:8px;">停止调度器</button>
                        <span id="scheduler-status-badge" class="badge badge-running" style="margin-left:12px;">运行中</span>
                    </div>
                </div>

                <h3 style="font-size:16px; margin-bottom:16px;">定时任务列表</h3>
                <table class="scheduler-table">
                    <thead>
                        <tr>
                            <th>任务ID</th>
                            <th>类型</th>
                            <th>配置</th>
                            <th>上次运行</th>
                            <th>下次运行</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody id="scheduler-jobs-body">
                    </tbody>
                </table>
                <div style="margin-top:16px;">
                    <button class="btn btn-small btn-secondary" onclick="refreshSchedulerStatus()">刷新</button>
                </div>
            </div>

            <div class="tab-content" id="tab-evaluate">
                <h3 style="font-size:16px; margin-bottom:16px;">离线评估指标</h3>
                <button class="btn" onclick="runEvaluation()">运行评估</button>
                <div id="evaluation-result" style="margin-top:16px;"></div>
            </div>
        </div>
    </div>

    <script>
        let currentRecItems = [];
        let trainingPollTimer = null;
        let schedulerPollTimer = null;

        function switchTab(tabName) {
            document.querySelectorAll('.tab-item').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector('[data-tab="' + tabName + '"]').classList.add('active');
            document.getElementById('tab-' + tabName).classList.add('active');
            if (tabName === 'scheduler') {
                refreshSchedulerStatus();
            }
        }

        document.querySelectorAll('.tab-item').forEach(tab => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        function setButtonLoading(btn, loadingText) {
            if (!btn) return;
            btn.dataset.originalText = btn.textContent;
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>' + loadingText;
        }

        function resetButton(btn) {
            if (!btn) return;
            btn.disabled = false;
            if (btn.dataset.originalText) {
                btn.textContent = btn.dataset.originalText;
            }
        }

        function updateTrainingStatusUI(status) {
            const box = document.getElementById('training-status');
            const badge = document.getElementById('ts-badge');
            const message = document.getElementById('ts-message');
            const progress = document.getElementById('ts-progress');

            box.classList.add('visible');
            badge.className = 'badge badge-' + status.status;
            progress.style.width = (status.progress || 0) + '%';

            if (status.status === 'idle') {
                badge.textContent = '空闲';
                message.textContent = status.message || '暂无训练任务';
            } else if (status.status === 'training') {
                badge.textContent = '训练中';
                message.textContent = status.message || '正在训练...';
            } else if (status.status === 'done') {
                badge.textContent = '完成';
                message.textContent = status.message || '训练完成';
            } else if (status.status === 'error') {
                badge.textContent = '错误';
                message.textContent = status.message || '训练出错';
            }
        }

        async function pollTrainingStatus(onComplete) {
            try {
                const res = await fetch('/api/training_status');
                const data = await res.json();
                updateTrainingStatusUI(data);

                if (data.status === 'training') {
                    trainingPollTimer = setTimeout(() => pollTrainingStatus(onComplete), 1000);
                } else {
                    if (trainingPollTimer) {
                        clearTimeout(trainingPollTimer);
                        trainingPollTimer = null;
                    }
                    if (onComplete) onComplete(data);
                }
            } catch(e) {
                console.error(e);
                if (trainingPollTimer) {
                    clearTimeout(trainingPollTimer);
                    trainingPollTimer = null;
                }
            }
        }

        async function loadStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('stat-users').textContent = data.n_users || 0;
                document.getElementById('stat-items').textContent = data.n_items || 0;
                document.getElementById('stat-interactions').textContent = data.n_interactions || 0;
                document.getElementById('stat-trained').textContent = data.is_trained ? '已训练' : '未训练';
            } catch(e) { console.error(e); }
        }

        async function getRecommendations() {
            const userId = document.getElementById('user-id').value;
            const count = document.getElementById('rec-count').value;
            const device = document.getElementById('context-device').value;

            const params = new URLSearchParams();
            params.append('user_id', userId);
            params.append('n', count);
            if (device) params.append('device', device);

            try {
                const res = await fetch('/api/recommend?' + params);
                const data = await res.json();
                currentRecItems = data.recommendations || [];
                renderRecommendations(data.recommendations || []);
            } catch(e) {
                document.getElementById('rec-list').innerHTML =
                    '<li style="color:#e53e3e; padding:20px;">获取推荐失败</li>';
            }
        }

        function renderRecommendations(items) {
            const list = document.getElementById('rec-list');
            if (!items || items.length === 0) {
                list.innerHTML = '<li style="color:#718096; text-align:center; padding:20px;">暂无推荐结果</li>';
                return;
            }

            list.innerHTML = items.map((item, idx) => {
                const channels = (item.channels && item.channels.length > 0)
                    ? item.channels.join(' + ')
                    : (item.channel || '-');
                return `
                <li class="recommendation-item">
                    <div class="item-rank">${idx + 1}</div>
                    <div class="item-info">
                        <div class="item-id">${item.item_id}</div>
                        <div class="item-meta">来源通道: ${channels}</div>
                        <div class="item-reason">${item.reason || ''}</div>
                    </div>
                    <div class="item-score">${item.score ? item.score.toFixed(4) : '-'}</div>
                </li>
            `}).join('');
        }

        async function simulateClick() {
            if (currentRecItems.length === 0) return;
            const item = currentRecItems[Math.floor(Math.random() * currentRecItems.length)];
            const userId = document.getElementById('user-id').value;

            try {
                await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: userId,
                        item_id: item.item_id,
                        channel: item.channel,
                        reward: 1.0
                    })
                });
                getRecommendations();
            } catch(e) { console.error(e); }
        }

        async function getSimilarItems() {
            const itemId = document.getElementById('item-id').value;
            const count = document.getElementById('similar-count').value;

            try {
                const res = await fetch(`/api/similar?item_id=${itemId}&n=${count}`);
                const data = await res.json();
                renderSimilarItems(data.similar_items || []);
            } catch(e) {
                document.getElementById('similar-list').innerHTML =
                    '<li style="color:#e53e3e; padding:20px;">获取相似物品失败</li>';
            }
        }

        function renderSimilarItems(items) {
            const list = document.getElementById('similar-list');
            if (!items || items.length === 0) {
                list.innerHTML = '<li style="color:#718096; text-align:center; padding:20px;">暂无相似物品</li>';
                return;
            }

            list.innerHTML = items.map((item, idx) => `
                <li class="recommendation-item">
                    <div class="item-rank">${idx + 1}</div>
                    <div class="item-info">
                        <div class="item-id">${item.item_id}</div>
                        <div class="item-meta">方法: ${item.method || '-'}</div>
                    </div>
                    <div class="item-score">${item.score ? item.score.toFixed(4) : '-'}</div>
                </li>
            `).join('');
        }

        function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            const dataType = document.getElementById('data-type').value;
            const mode = document.getElementById('import-mode').value;

            const reader = new FileReader();
            reader.onload = function(e) {
                uploadFileData(dataType, e.target.result, mode);
            };
            reader.readAsText(file);
        }

        async function uploadFileData(dataType, content, mode) {
            const btnSample = document.getElementById('btn-sample');
            const btnRetrain = document.getElementById('btn-retrain');
            setButtonLoading(btnSample, '训练中...');
            setButtonLoading(btnRetrain, '训练中...');
            document.getElementById('upload-result').innerHTML = '';

            try {
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        data_type: dataType,
                        content: content,
                        mode: mode
                    })
                });
                const data = await res.json();
                document.getElementById('upload-result').innerHTML =
                    '<div class="alert alert-info">数据上传成功，已开始后台训练，请稍候...</div>';

                pollTrainingStatus((finalStatus) => {
                    resetButton(btnSample);
                    resetButton(btnRetrain);
                    if (finalStatus.status === 'done') {
                        document.getElementById('upload-result').innerHTML =
                            '<div class="alert alert-success">' + finalStatus.message + '，共导入 ' + (data.count || 0) + ' 条记录</div>';
                    } else if (finalStatus.status === 'error') {
                        document.getElementById('upload-result').innerHTML =
                            '<div class="alert alert-error">' + finalStatus.message + '</div>';
                    }
                    loadStats();
                    loadChannelWeights();
                });
            } catch(e) {
                resetButton(btnSample);
                resetButton(btnRetrain);
                document.getElementById('upload-result').innerHTML =
                    '<div class="alert alert-error">上传失败</div>';
            }
        }

        async function loadSampleData() {
            const btnSample = document.getElementById('btn-sample');
            const btnRetrain = document.getElementById('btn-retrain');
            setButtonLoading(btnSample, '训练中...');
            setButtonLoading(btnRetrain, '训练中...');
            document.getElementById('action-result').innerHTML = '';

            try {
                const res = await fetch('/api/sample', { method: 'POST' });
                const data = await res.json();
                document.getElementById('action-result').innerHTML =
                    '<div class="alert alert-info">已开始后台训练，请稍候...</div>';

                pollTrainingStatus((finalStatus) => {
                    resetButton(btnSample);
                    resetButton(btnRetrain);
                    if (finalStatus.status === 'done') {
                        document.getElementById('action-result').innerHTML =
                            '<div class="alert alert-success">' + finalStatus.message + '</div>';
                    } else if (finalStatus.status === 'error') {
                        document.getElementById('action-result').innerHTML =
                            '<div class="alert alert-error">' + finalStatus.message + '</div>';
                    }
                    loadStats();
                    loadChannelWeights();
                });
            } catch(e) {
                resetButton(btnSample);
                resetButton(btnRetrain);
                document.getElementById('action-result').innerHTML =
                    '<div class="alert alert-error">加载失败</div>';
            }
        }

        async function retrainModel() {
            const btnSample = document.getElementById('btn-sample');
            const btnRetrain = document.getElementById('btn-retrain');
            setButtonLoading(btnSample, '训练中...');
            setButtonLoading(btnRetrain, '训练中...');
            document.getElementById('action-result').innerHTML = '';

            try {
                const res = await fetch('/api/retrain', { method: 'POST' });
                const data = await res.json();
                document.getElementById('action-result').innerHTML =
                    '<div class="alert alert-info">已开始后台训练，请稍候...</div>';

                pollTrainingStatus((finalStatus) => {
                    resetButton(btnSample);
                    resetButton(btnRetrain);
                    if (finalStatus.status === 'done') {
                        document.getElementById('action-result').innerHTML =
                            '<div class="alert alert-success">模型训练完成！</div>';
                    } else if (finalStatus.status === 'error') {
                        document.getElementById('action-result').innerHTML =
                            '<div class="alert alert-error">' + finalStatus.message + '</div>';
                    }
                    loadStats();
                });
            } catch(e) {
                resetButton(btnSample);
                resetButton(btnRetrain);
                document.getElementById('action-result').innerHTML =
                    '<div class="alert alert-error">训练失败</div>';
            }
        }

        async function loadChannelWeights() {
            try {
                const res = await fetch('/api/config/channels');
                const data = await res.json();
                const container = document.getElementById('channel-weights');

                container.innerHTML = Object.entries(data.weights || {}).map(([channel, weight]) => `
                    <div class="channel-weight">
                        <label>${channel}</label>
                        <input type="range" id="weight-${channel}" min="0" max="1" step="0.05"
                               value="${weight}" oninput="updateWeightDisplay('${channel}')">
                        <span id="weight-display-${channel}">${weight.toFixed(2)}</span>
                    </div>
                `).join('');
            } catch(e) { console.error(e); }
        }

        async function loadServerConfig() {
            try {
                const res = await fetch('/api/config/server');
                const data = await res.json();
                const container = document.getElementById('server-config-display');
                container.innerHTML = `
                    <div class="config-row">
                        <div class="config-key">Host</div>
                        <div class="config-value">${data.host}</div>
                    </div>
                    <div class="config-row">
                        <div class="config-key">Port</div>
                        <div class="config-value">${data.port}</div>
                    </div>
                    <div class="config-row">
                        <div class="config-key">Debug</div>
                        <div class="config-value">${data.debug ? '开启' : '关闭'}</div>
                    </div>
                    <div style="margin-top:8px; font-size:12px; color:#718096;">
                        提示：修改服务器配置需要重启服务
                    </div>
                `;
            } catch(e) { console.error(e); }
        }

        function updateWeightDisplay(channel) {
            const val = document.getElementById('weight-' + channel).value;
            document.getElementById('weight-display-' + channel).textContent = parseFloat(val).toFixed(2);
        }

        async function saveChannelWeights() {
            try {
                const res = await fetch('/api/config/channels');
                const data = await res.json();
                const weights = {};

                Object.keys(data.weights || {}).forEach(channel => {
                    const el = document.getElementById('weight-' + channel);
                    if (el) weights[channel] = parseFloat(el.value);
                });

                const saveRes = await fetch('/api/config/channels', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ weights })
                });
                const result = await saveRes.json();
                document.getElementById('config-result').innerHTML =
                    '<div class="alert alert-success">配置保存成功！</div>';
            } catch(e) {
                document.getElementById('config-result').innerHTML =
                    '<div class="alert alert-error">保存失败</div>';
            }
        }

        async function runEvaluation() {
            try {
                const res = await fetch('/api/evaluate');
                const data = await res.json();
                renderEvaluation(data.results || {});
            } catch(e) {
                document.getElementById('evaluation-result').innerHTML =
                    '<div class="alert alert-error">评估失败</div>';
            }
        }

        function renderEvaluation(results) {
            const container = document.getElementById('evaluation-result');
            const ks = Object.keys(results).map(Number).sort((a, b) => a - b);

            if (ks.length === 0) {
                container.innerHTML = '<div class="alert alert-error">暂无评估数据</div>';
                return;
            }

            const metrics = Object.keys(results[ks[0]]);

            let html = '<table class="evaluation-table"><thead><tr><th>指标</th>';
            ks.forEach(k => html += `<th>Top-${k}</th>`);
            html += '</tr></thead><tbody>';

            metrics.forEach(metric => {
                html += `<tr><td>${metric}</td>`;
                ks.forEach(k => {
                    const val = results[k][metric];
                    html += `<td>${val ? val.toFixed(4) : '-'}</td>`;
                });
                html += '</tr>';
            });

            html += '</tbody></table>';
            container.innerHTML = html;
        }

        async function refreshSchedulerStatus() {
            try {
                const res = await fetch('/api/scheduler/status');
                const data = await res.json();
                renderSchedulerStatus(data);
            } catch(e) { console.error(e); }
        }

        function renderSchedulerStatus(data) {
            const statusBadge = document.getElementById('scheduler-status-badge');
            statusBadge.textContent = data.running ? '运行中' : '已停止';
            statusBadge.className = 'badge badge-' + (data.running ? 'running' : 'stopped');

            const tbody = document.getElementById('scheduler-jobs-body');
            const jobs = data.jobs || {};
            const jobIds = Object.keys(jobs);

            if (jobIds.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#718096; padding:20px;">暂无调度任务</td></tr>';
                return;
            }

            const jobNames = {
                'incremental_update': '增量更新',
                'retrain': '全量重训练'
            };

            tbody.innerHTML = jobIds.map(jobId => {
                const job = jobs[jobId] || {};
                let configText = '';
                if (job.cron_expr) {
                    configText = `Cron: ${job.cron_expr}`;
                } else if (job.interval_seconds) {
                    const hours = Math.floor(job.interval_seconds / 3600);
                    const mins = Math.floor((job.interval_seconds % 3600) / 60);
                    if (hours > 0) {
                        configText = `间隔: ${hours}小时${mins > 0 ? mins + '分钟' : ''}`;
                    } else {
                        configText = `间隔: ${mins}分钟`;
                    }
                }
                return `
                    <tr>
                        <td><strong>${jobId}</strong><br><span style="font-size:12px; color:#718096;">${jobNames[jobId] || ''}</span></td>
                        <td>${job.cron_expr ? 'Cron定时' : '固定间隔'}</td>
                        <td style="font-family: monospace; font-size:12px;">${configText}</td>
                        <td>${job.last_run || '-'}</td>
                        <td>${job.next_run || '-'}</td>
                        <td><button class="btn btn-small btn-secondary" onclick="triggerJob('${jobId}')">立即执行</button></td>
                    </tr>
                `;
            }).join('');
        }

        async function triggerJob(jobId) {
            try {
                const res = await fetch(`/api/scheduler/trigger/${jobId}`, { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    alert('任务已触发');
                    refreshSchedulerStatus();
                    pollTrainingStatus(() => {
                        loadStats();
                        loadChannelWeights();
                    });
                    updateTrainingStatusUI({ status: 'training', message: `手动触发任务: ${jobId}`, progress: 10 });
                } else {
                    alert('触发失败: ' + (data.error || '未知错误'));
                }
            } catch(e) {
                alert('触发失败');
            }
        }

        async function startScheduler() {
            try {
                const res = await fetch('/api/scheduler/start', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    refreshSchedulerStatus();
                }
            } catch(e) { console.error(e); }
        }

        async function stopScheduler() {
            try {
                const res = await fetch('/api/scheduler/stop', { method: 'POST' });
                const data = await res.json();
                if (data.success) {
                    refreshSchedulerStatus();
                }
            } catch(e) { console.error(e); }
        }

        loadStats();
        loadChannelWeights();
        loadServerConfig();
        refreshSchedulerStatus();
        pollTrainingStatus();
    </script>
</body>
</html>
    """

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @app.route('/api/stats')
    def api_stats():
        return jsonify(rec_system.get_stats())

    @app.route('/api/recommend')
    def api_recommend():
        user_id = request.args.get('user_id', '')
        n = int(request.args.get('n', 10))
        device = request.args.get('device', None)

        context = {}
        if device:
            context['device'] = device

        recs = rec_system.recommend(user_id, n_items=n, context=context)
        return jsonify({
            'user_id': user_id,
            'recommendations': [
                {
                    'item_id': r.item_id,
                    'score': r.score,
                    'reason': r.reason,
                    'channel': r.channel,
                    'channels': r.channels if hasattr(r, 'channels') else [r.channel],
                    'rank': r.rank,
                } for r in recs
            ]
        })

    @app.route('/api/similar')
    def api_similar():
        item_id = request.args.get('item_id', '')
        n = int(request.args.get('n', 10))

        similar = rec_system.get_similar_items(item_id, n)
        return jsonify({
            'item_id': item_id,
            'similar_items': similar,
        })

    @app.route('/api/training_status', methods=['GET'])
    def api_training_status():
        return jsonify(training_status)

    @app.route('/api/upload', methods=['POST'])
    def api_upload():
        if training_status['status'] == 'training':
            return jsonify({'success': False, 'error': '已有训练任务正在进行中'}), 409

        data = request.json
        data_type = data.get('data_type', '')
        content = data.get('content', '')
        mode = data.get('mode', 'batch')

        loader = DataLoader(rec_system.data_dir)
        count = loader.upload_file(data_type, content, mode=mode)

        rec_system.load_data()

        job_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=_run_training_job,
            args=(job_id, rec_system.train_all, '数据上传后训练', rec_system),
            daemon=True
        )
        thread.start()

        return jsonify({'success': True, 'count': count, 'status': 'training', 'job_id': job_id})

    @app.route('/api/sample', methods=['POST'])
    def api_sample():
        if training_status['status'] == 'training':
            return jsonify({'success': False, 'error': '已有训练任务正在进行中'}), 409

        _generate_sample_data(rec_system.data_dir)
        rec_system.load_data()

        job_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=_run_training_job,
            args=(job_id, rec_system.train_all, '示例数据加载并训练', rec_system),
            daemon=True
        )
        thread.start()

        return jsonify({'success': True, 'status': 'training', 'job_id': job_id})

    @app.route('/api/retrain', methods=['POST'])
    def api_retrain():
        if training_status['status'] == 'training':
            return jsonify({'success': False, 'error': '已有训练任务正在进行中'}), 409

        job_id = str(uuid.uuid4())
        thread = threading.Thread(
            target=_run_training_job,
            args=(job_id, rec_system.retrain, '模型重训练', rec_system),
            daemon=True
        )
        thread.start()

        return jsonify({'success': True, 'status': 'training', 'job_id': job_id})

    @app.route('/api/feedback', methods=['POST'])
    def api_feedback():
        data = request.json
        user_id = data.get('user_id', '')
        item_id = data.get('item_id', '')
        channel = data.get('channel', '')
        reward = float(data.get('reward', 1.0))

        rec_system.record_feedback(user_id, item_id, reward, channel)
        return jsonify({'success': True})

    @app.route('/api/config/channels', methods=['GET'])
    def api_get_channels():
        weights = rec_system.multi_recall.get_channel_weights() if rec_system.multi_recall else {}
        return jsonify({'weights': weights})

    @app.route('/api/config/channels', methods=['POST'])
    def api_set_channels():
        data = request.json
        weights = data.get('weights', {})

        if rec_system.multi_recall:
            rec_system.multi_recall.update_channel_weights(weights)

        return jsonify({'success': True, 'weights': weights})

    @app.route('/api/config/server', methods=['GET'])
    def api_get_server_config():
        return jsonify({
            'host': config['server']['host'],
            'port': config['server']['port'],
            'debug': config['server']['debug'],
        })

    @app.route('/api/evaluate')
    def api_evaluate():
        from collections import defaultdict
        test_users = defaultdict(list)

        if rec_system.dataset.interactions is not None and len(rec_system.dataset.interactions) > 0:
            for _, row in rec_system.dataset.interactions.iterrows():
                uid = str(row['user_id'])
                iid = str(row['item_id'])
                test_users[uid].append(iid)

        test_users_dict = dict(list(test_users.items())[:50])
        results = rec_system.evaluate(test_users_dict)

        return jsonify({'results': results})

    @app.route('/api/users')
    def api_users():
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))

        users = []
        if rec_system.user_profile_builder:
            all_users = list(rec_system.user_profile_builder.profiles.values())
            start = (page - 1) * page_size
            users = [p.to_dict() for p in all_users[start:start + page_size]]

        return jsonify({
            'total': rec_system.dataset.n_users,
            'page': page,
            'page_size': page_size,
            'users': users,
        })

    @app.route('/api/items')
    def api_items():
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))

        items = []
        if rec_system.item_profile_builder:
            all_items = list(rec_system.item_profile_builder.profiles.values())
            start = (page - 1) * page_size
            items = [p.to_dict() for p in all_items[start:start + page_size]]

        return jsonify({
            'total': rec_system.dataset.n_items,
            'page': page,
            'page_size': page_size,
            'items': items,
        })

    @app.route('/api/scheduler/status', methods=['GET'])
    def api_scheduler_status():
        return jsonify({
            'running': scheduler._running,
            'jobs': scheduler.get_job_status(),
        })

    @app.route('/api/scheduler/trigger/<job_id>', methods=['POST'])
    def api_scheduler_trigger(job_id: str):
        try:
            scheduler.run_now(job_id)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/scheduler/start', methods=['POST'])
    def api_scheduler_start():
        scheduler.start()
        return jsonify({'success': True, 'running': scheduler._running})

    @app.route('/api/scheduler/stop', methods=['POST'])
    def api_scheduler_stop():
        scheduler.stop()
        return jsonify({'success': True, 'running': scheduler._running})

    return app


def _generate_sample_data(data_dir: str):
    import pandas as pd
    import os

    os.makedirs(data_dir, exist_ok=True)

    users_data = []
    for i in range(1, 51):
        interests_options = ['动作,科幻', '喜剧,爱情', '悬疑,犯罪', '动画,奇幻', '纪录片,音乐']
        genders = ['M', 'F']
        users_data.append({
            'user_id': f'user_{i}',
            'age': 18 + (i % 40),
            'gender': genders[i % 2],
            'interests': interests_options[i % len(interests_options)],
            'location': f'city_{(i % 10) + 1}',
        })
    pd.DataFrame(users_data).to_csv(os.path.join(data_dir, 'users.csv'), index=False)

    items_data = []
    categories = ['动作片', '喜剧片', '爱情片', '科幻片', '悬疑片', '动画片', '纪录片', '恐怖片']
    for i in range(1, 101):
        cat = categories[i % len(categories)]
        tags_options = ['热门,经典', '新上映,推荐', '高分,获奖', '小众,文艺']
        items_data.append({
            'item_id': f'item_{i}',
            'title': f'{cat}电影第{i}部',
            'category': cat,
            'tags': tags_options[i % len(tags_options)],
            'description': f'这是一部精彩的{cat}，讲述了一个动人的故事。',
            'popularity': 100 - i,
        })
    pd.DataFrame(items_data).to_csv(os.path.join(data_dir, 'items.csv'), index=False)

    import numpy as np
    np.random.seed(42)
    interactions_data = []
    for i in range(1000):
        user_idx = np.random.randint(1, 51)
        item_idx = np.random.randint(1, 101)
        rating = round(3 + np.random.rand() * 2, 1)
        days_ago = np.random.randint(0, 90)
        from datetime import datetime, timedelta
        ts = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        interactions_data.append({
            'user_id': f'user_{user_idx}',
            'item_id': f'item_{item_idx}',
            'rating': rating,
            'timestamp': ts,
            'behavior_type': 'view' if np.random.rand() > 0.3 else 'like',
        })
    pd.DataFrame(interactions_data).to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)


def run_server(host: str = None, port: int = None, debug: bool = None,
               data_dir: str = None):
    if host is None:
        host = config['server']['host']
    if port is None:
        port = config['server']['port']
    if debug is None:
        debug = config['server']['debug']
    if data_dir is None:
        data_dir = config['data']['data_dir']

    app = create_app(data_dir=data_dir)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server()
