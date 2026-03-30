import { defineSiteConfig } from 'valaxy'

export default defineSiteConfig({
  url: 'https://whispering-dust.github.io/',
  lang: 'zh-CN',
  title: '落尘の小站',
  author: {
    name: '落尘丿叶知秋',
    avatar: 'https://whispering-dust.github.io/images/avatar.jpg',
  },
  description: 'Stay hungry, stay foolish.',
  favicon: 'https://whispering-dust.github.io/favicon.ico',
  social: [
    {
      name: 'RSS',
      link: '/atom.xml',
      icon: 'i-ri-rss-line',
      color: 'orange',
    },
    {
      name: 'GitHub',
      link: 'https://github.com/YunYouJun',
      icon: 'i-ri-github-line',
      color: '#6e5494',
    },
    {
      name: '知乎',
      link: 'https://www.zhihu.com/people/yunyoujun/',
      icon: 'i-ri-zhihu-line',
      color: '#0084FF',
    },
    {
      name: '哔哩哔哩',
      link: 'https://space.bilibili.com/1579790',
      icon: 'i-ri-bilibili-line',
      color: '#FF8EB3',
    },
    {
      name: 'E-Mail',
      link: 'mailto:me@yunyoujun.cn',
      icon: 'i-ri-mail-line',
      color: '#8E71C1',
    },
  ],

  search: {
    enable: false,
  },

  license: {
    // 结尾版权许可证信息
    enabled: false,
  },

  statistics: {
    // 阅读时间
      enable: true,
      readTime: {
        speed: {
          cn: 300,
          en: 200,
        },
      },
  },
  sponsor: {
    enable: true,
    title: '我很可爱，请给我钱！',
    methods: [
      {
        name: '支付宝',
        url: '',
        color: '#00A3EE',
        icon: 'i-ri-alipay-line',
      },
      {
        name: '微信支付',
        url: '',
        color: '#2DC100',
        icon: 'i-ri-wechat-pay-line',
      },
    ],
  },
})
