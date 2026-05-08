import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import {
  BarChart,
  GaugeChart,
  HeatmapChart,
  LineChart,
  PieChart,
  RadarChart,
  ScatterChart,
} from 'echarts/charts'
import {
  DataZoomComponent,
  DatasetComponent,
  GraphicComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  MarkPointComponent,
  PolarComponent,
  RadarComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
  TransformComponent,
  VisualMapComponent,
} from 'echarts/components'
import { LabelLayout, UniversalTransition } from 'echarts/features'

/**
 * 项目级按需注册 ECharts 模块；若某图表报「Component ... is used but not imported」，
 * 将对应 chart/component 按官方文档加入此处。
 */
export function setupEcharts(): void {
  use([
    CanvasRenderer,
    BarChart,
    LineChart,
    PieChart,
    ScatterChart,
    RadarChart,
    HeatmapChart,
    GaugeChart,
    TitleComponent,
    TooltipComponent,
    LegendComponent,
    GridComponent,
    PolarComponent,
    RadarComponent,
    DatasetComponent,
    ToolboxComponent,
    DataZoomComponent,
    VisualMapComponent,
    GraphicComponent,
    MarkLineComponent,
    MarkPointComponent,
    TransformComponent,
    LabelLayout,
    UniversalTransition,
  ])
}
