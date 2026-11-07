"""``python -m ai_search_plugin [list | plan <slime|verl> [components] [key=value ...]]``"""
import sys

from . import list_components, plan


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv.pop(0) if argv else "list"
    if cmd == "list":
        for c in list_components():
            print(f"{c.name:18s} {c.category:15s} {c.status:7s} default={'on ' if c.default else 'off'} "
                  f"{','.join(c.frameworks):10s} {c.summary}")
        return 0
    if cmd == "plan":
        if not argv:
            print("usage: python -m ai_search_plugin plan <slime|verl> [components] [key=value ...]", file=sys.stderr)
            return 2
        fw = argv.pop(0)
        comps = argv.pop(0) if argv and "=" not in argv[0] else "default"
        opts = dict(a.split("=", 1) for a in argv)
        print(plan(fw, comps, opts).shell())
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
