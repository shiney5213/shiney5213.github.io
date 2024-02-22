---
layout: page
title: 하버드 회복탄력성 수업
description: >
  How you install Hydejack depends on whether you start a new site,
  or change the theme of an existing site.
hide_description: true
sitemap: false
---

<img src="/assets/img/books/2024-01-resilience-in-harvard.jpg" width="200px" height="300px" title="book-2024-01">

0. this unordered seed list will be replaced by toc as unordered list
{:toc}


## 읽게 된 계기
며칠 전 계획했던 일정이 연기되는 일이 있었다. <br>
나에게는 굉장히 중요한 일이었고, 앞으로 일정에 차질이 생겨 난감했고, 화가 났다.<br>
문제는 내가 내 기분을 컨트롤 하지 못하였고 부정적인 생각만 하면서, 주변 사람들에게 짜증을 내고, 힘들게 하였다는 것이다.<br>
아무리 내가 MBTI가 J라고 하더라도, <br>
세상 일이 내 마음대로 되지 않는다는 것에 대해 잘 알고있음에도 불구하고, <br>
나에게, 또 주변 사람에게 심하게 행동한 내가 너무 실망스러웠다.<br>
그 동안 내 안의 정해진 틀을 만들어 놓고, 그 틀에 모든 일을 끼워맞추려고 했던 것 같다.<br>
그 틀에 맞지 않으면 화를 내고 짜증을 내고, 결국 그 틀 안으로 들어가 문을 닫아버렸다.<br>
뭔가 변화가 필요하다. <br>
그래서 유연한 사고에 대해 관심을 갖게 되었다.<br>
이 책을 통해<br>
오픈 마인드로, 다른 사람의 의견을 존중하고,<br>
다양한 시각으로 사건을 바라보고, 긍정적으로 사고할 수 있기를 바란다.<br>
또한 유연하면서도 단단한 마음가짐으로, 변화무쌍한 이 세상을 굳건히 잘 헤쳐날수 있었으면 좋겠다. <br>


## 1장 누구에게나 회복탄력성은 있다
## 2장 대인관계(Connections)
## 3장 유연성(Flexibility)
## 4장 끈기(Perseverance)
## 5장 자기조절(Self-Regulation)
## 6장 긍정성(Positivity)
## 7장 자기돌봄(Self-Care)
## 8장 8장 회복탄력성은 마라톤이다
For new sites, the best way to get started with Hydejack is via the Starter Kit. 
It comes with a documented config file and example content that gets you started quickly.

If you have a GitHub account, fork the [Hydejack Starter Kit][hsc] repository. 
Otherwise [download the Starter Kit][src] and unzip them somewhere on your machine.

If you bought the __PRO Version__ of Hydejack, use the contents of the `starter-kit` folder instead.

In addition to the docs here, you can follow the quick start guide in the Starter Kit.
{:.note}

You can now jump to [running locally](#running-locally).

You can now also [![Deploy to Netlify][dtn]][nfy]{:.no-mark-external} directly.
{:.note}

[hsc]: https://github.com/hydecorp/hydejack-starter-kit
[src]: https://github.com/hydecorp/hydejack-starter-kit/archive/v9.1.6.zip
[nfy]: https://app.netlify.com/start/deploy?repository=https://github.com/hydecorp/hydejack-starter-kit
[dtn]: https://www.netlify.com/img/deploy/button.svg


## Existing sites
If you have an existing site that you'd like to upgrade to Hydejack you can install the theme via bundler.
Add the following to your `Gemfile`:

~~~ruby
# file: `Gemfile`
gem "jekyll-theme-hydejack"
~~~

If you bought the __PRO Version__ of Hydejack, copy the `#jekyll-theme-hydejack` folder into the root folder of your site,
and add the following to your `Gemfile` instead:

~~~ruby
# file: `Gemfile`
gem "jekyll-theme-hydejack", path: "./#jekyll-theme-hydejack"
~~~

The folder is prefixed with a `#` to indicate that this folder is different from regular Jekyll content. 
The `#` char was choosen specifically because it is on of the four characters ignored by Jekyll by default (`.`, `_` , `#`, `~`).
{:.note}

In your config file, change the `theme` to Hydejack:

~~~yml
# file: `_config.yml`
theme: jekyll-theme-hydejack
~~~

Hydejack comes with a default configuration file that takes care most of the configuration,
but it pays off to check out the example config file in the Starter Kit to see what's available.

You can now jump to [running locally](#running-locally).

### Troubleshooting
If your existing site combines theme files with your content (as did previous verisons of Hydejack/PRO),
make sure to delete the following folders:

- `_layouts`
- `_includes` 
- `_sass` 
- `assets`

The `assets` folder most likely includes theme files as well as your personal/content files. 
Make sure to only delete files that belong to the old theme!


## GitHub Pages
If you want to build your site on [GitHub Pages][ghp], check out the [`gh-pages` branch][gpb] in the Hydejack Starter Kit repo.

[ghp]: https://jekyllrb.com/docs/github-pages/
[gpb]: https://github.com/hydecorp/hydejack-starter-kit/tree/gh-pages

For existing sites, you can instead set the `remote_theme` key as follows:

```yml
# file: `_config.yml`
remote_theme: hydecorp/hydejack@v9.1.6
```

Make sure the `plugins` list contains `jekyll-include-cache` (create if it doesn't exist):
{:.note title="Important"}

```yml
# file: `_config.yml`
plugins:
  - jekyll-include-cache
```

To run this configuration locally, make sure the following is part of your `Gemfile`:

```ruby
# file: `Gemfile`
gem "github-pages", group: :jekyll_plugins
gem "jekyll-include-cache", group: :jekyll_plugins
```

Note that Hydejack has a reduced feature set when built on GitHub Pages. 
Specifically, using KaTeX math formulas doesn't work when built in this way.
{:.note}


## Running locally
Make sure you've `cd`ed into the directory where `_config.yml` is located.
Before running for the first time, dependencies need to be fetched from [RubyGems](https://rubygems.org/):

~~~bash
$ bundle install
~~~

If you are missing the `bundle` command, you can install Bundler by running `gem install bundler`.
{:.note}

Now you can run Jekyll on your local machine:

~~~bash
$ bundle exec jekyll serve
~~~

and point your browser to <http://localhost:4000> to see Hydejack in action.


Continue with [Config](config.md){:.heading.flip-title}
{:.read-more}


[upgrade]: upgrade.md
