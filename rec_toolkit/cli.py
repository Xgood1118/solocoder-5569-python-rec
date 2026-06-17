import click
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rec_toolkit.config import config
from rec_toolkit.data import DataLoader, Dataset
from rec_toolkit.recommender import RecommenderSystem
from rec_toolkit.evaluation import Evaluator
from rec_toolkit.association import AssociationRuleMiner


@click.group()
@click.option('--data-dir', default='data', help='数据目录')
@click.option('--config', 'config_file', default=None, help='配置文件路径')
@click.pass_context
def cli(ctx, data_dir, config_file):
    ctx.ensure_object(dict)
    ctx.obj['data_dir'] = data_dir

    if config_file and os.path.exists(config_file):
        config.load_yaml(config_file)


@cli.command()
@click.option('--host', default='0.0.0.0', help='服务器地址')
@click.option('--port', default=5000, help='端口号', type=int)
@click.option('--debug/--no-debug', default=True, help='调试模式')
@click.pass_context
def serve(ctx, host, port, debug):
    from rec_toolkit.web import run_server
    click.echo(f'启动推荐系统 Web 服务: {host}:{port}')
    run_server(host=host, port=port, debug=debug, data_dir=ctx.obj['data_dir'])


@cli.command()
@click.option('--user-id', required=True, help='用户ID')
@click.option('--n', default=10, help='推荐数量', type=int)
@click.pass_context
def recommend(ctx, user_id, n):
    rec = RecommenderSystem(data_dir=ctx.obj['data_dir'])
    rec.train_all()

    recs = rec.recommend(user_id, n_items=n)
    click.echo(f'\n为用户 {user_id} 推荐 Top-{n}:\n')
    for i, r in enumerate(recs, 1):
        click.echo(f'  {i:2d}. {r.item_id}  分数: {r.score:.4f}  理由: {r.reason}')


@cli.command()
@click.option('--item-id', required=True, help='物品ID')
@click.option('--n', default=10, help='相似数量', type=int)
@click.pass_context
def similar(ctx, item_id, n):
    rec = RecommenderSystem(data_dir=ctx.obj['data_dir'])
    rec.train_all()

    items = rec.get_similar_items(item_id, n)
    click.echo(f'\n与 {item_id} 相似的 Top-{n}:\n')
    for i, item in enumerate(items, 1):
        click.echo(f'  {i:2d}. {item["item_id"]}  分数: {item["score"]:.4f}  方法: {item["method"]}')


@cli.command()
@click.argument('data_type', type=click.Choice(['users', 'items', 'interactions']))
@click.argument('csv_file', type=click.Path(exists=True))
@click.option('--mode', type=click.Choice(['batch', 'incremental']), default='batch',
              help='导入模式')
@click.pass_context
def import_data(ctx, data_type, csv_file, mode):
    loader = DataLoader(ctx.obj['data_dir'])

    with open(csv_file, 'r', encoding='utf-8') as f:
        content = f.read()

    result = loader.validate_csv_content(data_type, content)
    if not result['valid']:
        click.echo('CSV 校验失败:')
        for err in result['errors']:
            click.echo(f'  - {err}')
        sys.exit(1)

    count = loader.upload_file(data_type, content, mode=mode)
    click.echo(f'成功导入 {count} 条 {data_type} 数据')


@cli.command()
@click.argument('data_type', type=click.Choice(['users', 'items', 'interactions']))
@click.argument('csv_file', type=click.Path())
@click.pass_context
def validate(ctx, data_type, csv_file):
    loader = DataLoader(ctx.obj['data_dir'])

    with open(csv_file, 'r', encoding='utf-8') as f:
        content = f.read()

    result = loader.validate_csv_content(data_type, content)

    if result['valid']:
        click.echo(click.style('✓ CSV 校验通过', fg='green'))
        click.echo(f'  行数: {result["row_count"]}')
        click.echo(f'  列: {", ".join(result["columns"])}')
    else:
        click.echo(click.style('✗ CSV 校验失败', fg='red'))
        for err in result['errors']:
            click.echo(f'  - {err}')
        sys.exit(1)


@cli.command()
@click.pass_context
def train(ctx):
    click.echo('开始训练推荐模型...')
    rec = RecommenderSystem(data_dir=ctx.obj['data_dir'])
    rec.train_all()
    stats = rec.get_stats()
    click.echo(click.style('训练完成!', fg='green'))
    click.echo(f'  用户数: {stats["n_users"]}')
    click.echo(f'  物品数: {stats["n_items"]}')
    click.echo(f'  交互数: {stats["n_interactions"]}')


@cli.command()
@click.pass_context
def evaluate(ctx):
    click.echo('运行离线评估...')
    rec = RecommenderSystem(data_dir=ctx.obj['data_dir'])
    rec.train_all()

    from collections import defaultdict
    test_users = defaultdict(list)
    dataset = rec.dataset
    for _, row in dataset.interactions.iterrows():
        uid = str(row['user_id'])
        iid = str(row['item_id'])
        test_users[uid].append(iid)

    test_users_dict = dict(list(test_users.items())[:50])
    results = rec.evaluate(test_users_dict)

    evaluator = Evaluator()
    click.echo('\n' + evaluator.format_results(results))


@cli.command()
@click.option('--min-support', default=0.01, type=float, help='最小支持度')
@click.option('--min-confidence', default=0.5, type=float, help='最小置信度')
@click.option('--algorithm', type=click.Choice(['apriori', 'fpgrowth']),
              default='apriori', help='算法')
@click.option('--output', default=None, help='输出文件')
@click.pass_context
def association(ctx, min_support, min_confidence, algorithm, output):
    click.echo(f'使用 {algorithm} 挖掘关联规则...')
    dataset = Dataset(ctx.obj['data_dir'])
    dataset.load_all()

    output_file = output or os.path.join(ctx.obj['data_dir'], 'association_rules.txt')
    miner = AssociationRuleMiner(
        algorithm=algorithm,
        min_support=min_support,
        min_confidence=min_confidence,
        output_file=output_file,
    )
    miner.fit_from_interactions(dataset.interactions)

    click.echo(click.style(f'挖掘完成! 共 {len(miner.rules)} 条规则', fg='green'))
    click.echo(f'结果已保存到: {output_file}')

    if miner.rules:
        click.echo('\nTop 10 规则:')
        for i, rule in enumerate(miner.rules[:10], 1):
            ante = ', '.join(sorted(rule.antecedent))
            cons = ', '.join(sorted(rule.consequent))
            click.echo(f'  {i:2d}. {{{ante}}} => {{{cons}}}  '
                       f'支持度: {rule.support:.4f}  '
                       f'置信度: {rule.confidence:.4f}  '
                       f'提升度: {rule.lift:.4f}')


@cli.command()
@click.pass_context
def stats(ctx):
    dataset = Dataset(ctx.obj['data_dir'])
    dataset.load_all()

    click.echo('\n数据统计:')
    click.echo(f'  用户数: {dataset.n_users}')
    click.echo(f'  物品数: {dataset.n_items}')
    click.echo(f'  交互数: {len(dataset.interactions)}')

    if len(dataset.interactions) > 0 and 'rating' in dataset.interactions.columns:
        click.echo(f'  平均评分: {dataset.interactions["rating"].mean():.2f}')

    click.echo(f'\n数据目录: {ctx.obj["data_dir"]}')


@cli.command()
@click.option('--n-users', default=50, type=int, help='用户数量')
@click.option('--n-items', default=100, type=int, help='物品数量')
@click.option('--n-interactions', default=1000, type=int, help='交互数量')
@click.pass_context
def sample(ctx, n_users, n_items, n_interactions):
    from rec_toolkit.web.app import _generate_sample_data

    os.makedirs(ctx.obj['data_dir'], exist_ok=True)
    _generate_sample_data(ctx.obj['data_dir'])

    click.echo(click.style('示例数据生成完成!', fg='green'))
    click.echo(f'  数据目录: {ctx.obj["data_dir"]}')
    click.echo(f'  用户: {n_users}, 物品: {n_items}, 交互: {n_interactions}')


def main():
    cli(obj={})


if __name__ == '__main__':
    main()
