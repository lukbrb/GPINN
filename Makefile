dehnen:
	@python3 run_dehnen.py

test:
	@echo "test suite is not implemented yet"

PLOTSAVED = True

plots:
	@python3 plots.py $(PLOTSAVED)
