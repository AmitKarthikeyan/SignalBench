"""initial schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.create_table('experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('git_commit', sa.String(length=40), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('datasets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('symbol', sa.String(length=20), nullable=False),
        sa.Column('start_date', sa.String(length=10), nullable=False),
        sa.Column('end_date', sa.String(length=10), nullable=False),
        sa.Column('frequency', sa.String(length=10), nullable=False),
        sa.Column('features_version', sa.String(length=50), nullable=False),
        sa.Column('data_hash', sa.String(length=64), nullable=True),
        sa.Column('feature_hash', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_datasets_symbol', 'datasets', ['symbol'])
    
    op.create_table('model_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=False),
        sa.Column('training_settings', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column('model_config_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', name='runstatus'), nullable=False),
        sa.Column('train_started_at', sa.DateTime(), nullable=True),
        sa.Column('train_finished_at', sa.DateTime(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('train_rows', sa.Integer(), nullable=True),
        sa.Column('test_rows', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id'], ),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
        sa.ForeignKeyConstraint(['model_config_id'], ['model_configs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_runs_experiment_id', 'runs', ['experiment_id'])
    op.create_index('ix_runs_dataset_id', 'runs', ['dataset_id'])
    op.create_index('ix_runs_model_config_id', 'runs', ['model_config_id'])
    
    op.create_table('metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('roc_auc', sa.Float(), nullable=True),
        sa.Column('pr_auc', sa.Float(), nullable=True),
        sa.Column('brier_score', sa.Float(), nullable=True),
        sa.Column('ece', sa.Float(), nullable=True),
        sa.Column('confusion_matrix', sa.JSON(), nullable=True),
        sa.Column('train_time_seconds', sa.Float(), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=True),
        sa.Column('extra_metrics', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_metrics_run_id', 'metrics', ['run_id'])

def downgrade() -> None:
    op.drop_index('ix_metrics_run_id', table_name='metrics')
    op.drop_table('metrics')
    op.drop_index('ix_runs_model_config_id', table_name='runs')
    op.drop_index('ix_runs_dataset_id', table_name='runs')
    op.drop_index('ix_runs_experiment_id', table_name='runs')
    op.drop_table('runs')
    op.drop_table('model_configs')
    op.drop_index('ix_datasets_symbol', table_name='datasets')
    op.drop_table('datasets')
    op.drop_table('experiments')
