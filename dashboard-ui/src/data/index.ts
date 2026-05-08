/**
 * 仪表盘静态数据的唯一对外入口。
 *
 * - PLATFORMS：平台列表 → 侧栏品牌筛选、卡片/对比中的平台信息、详情当前平台
 * - DIMS：测试维度与顶栏筛选 pills → 维度切换、筛选状态
 * - CARD_DATA：各维度概览卡片 → 概览网格、对比 KPI
 * - OP_TABLE / INFER_TABLE / TRAIN_TABLE / COMM_TABLE / BW_TABLE：各维度详情表与图表数据源
 * - DATA_UPDATED_AT：侧栏数据更新日期
 * - CI_SUMMARY / CI_CHART_LABELS / CI_SERIES_BY_DIM / getCiSeriesForDim：详情 CI 区 KPI 与折线图
 * - LOGO_*：顶栏 logo（data URI）
 *
 * 业务与展示层请只从此模块 import（`@/data`），勿直连子路径以免散落引用。
 */
export * from './dashboardConfig'
export * from './logoDataUri'
