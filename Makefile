dehnen:
	@python3 code/run_dehnen.py

expdisc:
	@python3 code/run_disc.py
test:
	@echo "test suite is not implemented yet"

PLOTSAVED = True

plots:
	@python3 code/plots.py $(PLOTSAVED)
