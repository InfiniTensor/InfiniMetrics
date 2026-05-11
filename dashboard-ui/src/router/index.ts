import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/views/dashboard/DashboardView.vue'

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'overview',
      component: DashboardView,
    },
    {
      path: '/detail/:platKey/:dimKey',
      name: 'detail',
      component: DashboardView,
    },
    {
      path: '/compare',
      name: 'compare',
      component: DashboardView,
    },
  ],
})
