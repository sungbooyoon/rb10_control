import rbpodo as rb

def _main():
    robot = rb.Cobot("10.0.2.7")
    rc = rb.ResponseCollector()

    res = robot.set_freedrive_mode(rc, True)
    print("freedrive ON res:", res)
	
if __name__ == "__main__":
    _main()