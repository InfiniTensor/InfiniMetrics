import { createApp } from 'vue'
import 'ant-design-vue/dist/reset.css'
import './style.css'
import './styles/dashboard.css'
import { setupEcharts } from './plugins/echarts'
import App from './App.vue'
import { router } from './router'

setupEcharts()

createApp(App).use(router).mount('#app')
