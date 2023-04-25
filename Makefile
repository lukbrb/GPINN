dehnen:
	@python3 run_dehnen.py

expdisc:
	@python3 run_disc.py
test:
	@echo "test suite is not implemented yet"

PLOTSAVED = True

plots:
	@python3 plots.py $(PLOTSAVED)
