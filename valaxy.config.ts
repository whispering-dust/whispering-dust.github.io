import type { UserThemeConfig } from 'valaxy-theme-yun'
import { defineValaxyConfig } from 'valaxy'

// add icons what you will need
const safelist = [
  'i-ri-home-line',
]

/**
 * User Config
 */
export default defineValaxyConfig<UserThemeConfig>({
  // site config see site.config.ts

  theme: 'yun',

  themeConfig: {
    banner: {
      enable: true,
      title: '落尘の小站',
    },

    // 背景图（public/images/ 下的图片）
    bg_image: {
      enable: true,
      url: '/images/bg.jpg',
      dark: '/images/bg_dark.jpg',
      opacity: 0.85,
    },

    // 自定义语录
    // say: {
    //   enable: true,
    //   api: '/say.json',
    //   hitokoto: { enable: false },
    // },

    // 公告
    notice: {
      enable: true,
      content: '欢迎来到落尘の小站！',
    },

    nav:[
      {
        text: '分类',
        link: '/categories/',
        icon: 'i-ri-apps-line',
      },
      {
        text: '标签',
        link: '/tags/',
        icon: 'i-ri-bookmark-3-line',
      },
      {
        text: '友链',
        link: '/links/',
        icon: 'i-ri-open-arm-line',
      },
    ],
    pages: [
      {
        name: '分类',
        url: '/categories/',
        icon: 'i-ri-apps-line',
        color: 'dodgerblue',
      },
      {
        name: '标签',
        url: '/tags/',
        icon: 'i-ri-bookmark-3-line',
        color: 'dodgerblue',
      },
      {
        name: '友链',
        url: '/links/',
        icon: 'i-ri-open-arm-line',
        color: 'hotpink',
      },
    ],

    footer: {
      since: 2016,
      beian: {
        enable: true,
        icp: '',
        police: '',
      },
    },
  },

  unocss: { safelist },
})
