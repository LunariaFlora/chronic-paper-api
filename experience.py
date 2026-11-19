from chronic.console import main

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,RuntimeError,TypeError) as e:raise SystemExit(f'未完成：{e}')
