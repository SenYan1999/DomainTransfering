import multiprocessing as mp
import re
import os

from bs4 import BeautifulSoup
from urllib.request import urlopen

industry2code = {'Petrochemical': 400115939,
                 'FundamentalChemical': 400115930}
ipo_dir = 'data/IPO'
cpu_count = mp.cpu_count()

def remove_specific_char(text):
    return text.replace('[', '').replace(']', '').replace('"', '')

def get_stock_info_jobs(page, code):
    url = 'http://q.jrjimg.cn/?q=cn|s|sa,bk%s&c=m&n=hqa&o=pl,d&p=%d020&ib=%d&_dc=1595301735179' % (code, page, code)
    html = urlopen(url)
    lines = html.readlines()
    stocks = {}

    stocks_info = [line.decode('gbk') for line in lines[4:-1]]
    for stock_info in stocks_info:
        stock_info = remove_specific_char(stock_info.strip()).split(',')
        stocks[int(stock_info[1])] = stock_info[2]

    print('Page %d done.' % page)
    return stocks

def parse_html_job(path):
    with open(path, 'rb') as f:
        html = f.read()

    soup = BeautifulSoup(html, features='lxml')
    paragraph = soup.find_all('p', {'class': 'p1'})
    assert len(paragraph) == 1
    paragraph = paragraph[0]

    lines = paragraph.text.split('\u2002')

    save_path = path.replace('html', 'txt')
    with open(save_path, 'w') as f:
        for line in lines:
            if line != '':
                f.write(line)
                f.write('\n')

    print('Parse txt to: %s' % save_path)

def get_ipo_job(code, name, industry):
    url = 'http://stock.jrj.com.cn/share,%d,ssggs.shtml' % code
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')
    all_ipos = soup.find_all('a', {'target': '_blank'})

    base_url = 'http://stock.jrj.com.cn/'

    for i, ipo in enumerate(all_ipos):
        assert '上市公告书' in ipo.text
        ipo_url = base_url + ipo['href']

        ipo_html = urlopen(ipo_url).read()
        save_path = os.path.join(ipo_dir, '%s/html/%s_%s_%d.html' % (industry, str(code), name, i))

        with open(save_path, 'wb') as f:
            f.write(ipo_html)

        print('Save html to: %s' % save_path)

def get_industry_stocks(industry):
    # get the total pages
    page = 1
    code = industry2code[industry]

    url = 'http://q.jrjimg.cn/?q=cn|s|sa,bk%s&c=m&n=hqa&o=pl,d&p=%d020&ib=%d&_dc=1595301735179' % (code, page, code)
    html = urlopen(url)
    lines = html.readlines()

    pages_regex = re.compile(r'pages:(\d+),')
    pages = int((pages_regex.findall(lines[1].decode('gbk'))[0]))

    # get all stocks of the industry
    total_stocks = []
    pool = mp.Pool(processes=cpu_count)
    for page in range(1, pages+1):
        result = pool.apply_async(get_stock_info_jobs, [page, code])
        total_stocks.append(result)
    pool.close()
    pool.join()

    stocks = {}
    for stock in total_stocks:
        stocks.update(stock.get())

    return stocks

def save_ipo_html(stocks, industry):
    pool = mp.Pool(processes=cpu_count)
    for code, name in stocks.items():
        pool.apply_async(get_ipo_job, (code, name, industry,))
    pool.close()
    pool.join()

def parse_ipo_html(industry):
    html_path = os.path.join(ipo_dir, '%s/html' % industry)
    files = os.listdir(html_path)

    pool = mp.Pool(processes=cpu_count)
    for file in files:
        pool.apply_async(parse_html_job, (os.path.join(html_path, file),))
    pool.close()
    pool.join()

if __name__ == '__main__':
    for industry, code in industry2code.items():
        stocks = get_industry_stocks(industry)
        save_ipo_html(stocks, industry)
        parse_ipo_html(industry)

