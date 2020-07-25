import requests
import json


class Spider:
    def __init__(self):
        self.url = 'https://baike.baidu.com/wikitag/taglist?tagId=76613'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Host': 'baike.baidu.com'
        }

    def run(self):
        page = 0
        all_item = []
        while True:
            ret = self.get_next(page)
            all_item.extend([item.get('lemmaTitle') for item in ret.get('lemmaList')])
            print(ret.get('page'))
            if int(ret.get('totalPage')) <= page:
                break
            page += 1
        with open('data.json', 'w', encoding='utf-8-sig', newline='') as f:
            json.dump(all_item, f, indent='    ', ensure_ascii=False)

    def get_next(self, page):
        params = {
            'limit': 24,
            'timeout': 3000,
            'filterTags': [],
            'tagId': 76613,
            'fromLemma': False,
            'contentLength': 40,
            'page': str(page)
        }
        ret = requests.post(url='https://baike.baidu.com/wikitag/api/getlemmas',
                            data=params, headers=self.headers)
        return ret.json()


if __name__ == '__main__':
    Spider().run()