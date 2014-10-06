from __future__ import print_function, division

import web
import argparse
import json
import subprocess
import sys

import ll


urls = (
        '/(.*)', 'hello'
        )


class hello:
    def _dostuff(self, user_data):
        if not user_data:
            return 'type your json request into the url'
        try:
            result = json.dumps(ll.process_json_in(json.loads(user_data)))
        except Exception as e:
            result = str(e)
        return result
    def GET(self, user_data):
        return self._dostuff(user_data)
    def POST(self, user_data):
        i = web.input()
        data = web.data()
        return self._dostuff(data)


def main(args):
    app = web.application(urls, dict(hello=hello))
    app.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
