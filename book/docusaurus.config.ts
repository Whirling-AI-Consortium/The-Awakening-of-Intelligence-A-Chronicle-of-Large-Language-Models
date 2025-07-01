import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: '智能的觉醒', // 替换为你的书籍标题
  tagline: '大语言模型发展编年史', // 替换为你的书籍副标题
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://whirling-ai-consortium.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'whirling-ai-consortium', // 通常是你的 GitHub 组织或用户名
  projectName: 'The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models', // 通常是你的仓库名

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/whirling-ai-consortium/The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models/edit/main/', // 修改为你的仓库编辑链接
            remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: '主页', // 替换为你的书籍标题
      logo: {
        alt: 'My Book Logo', // 替换为你的书籍 Logo Alt Text
        src: 'img/logo.svg', // 替换为你的书籍 Logo 路径
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar', // 保持与你的 sidebars.ts 中定义的 sidebarId 一致
          position: 'left',
          label: 'Read Book', // 修改导航栏标签
        },
        {
          href: 'https://github.com/whirling-ai-consortium/The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models', // 修改为你的 GitHub 仓库链接
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Read',
          items: [
            {
              label: 'Introduction', // 修改为你的书籍起始章节
              to: '/docs/intro', // 修改为你的书籍起始章节路径
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions', // 可以修改为你的协作讨论平台
              href: 'https://github.com/whirling-ai-consortium/The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models/discussions', // 修改为你的仓库 Discussions 链接
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/whirling-ai-consortium/The-Awakening-of-Intelligence-A-Chronicle-of-Large-Language-Models', // 修改为你的 GitHub 仓库链接
              remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Whirling AI Consortium. Built with Docusaurus.`, // 修改版权信息
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;