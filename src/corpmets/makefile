# build automation

MTARG=		target

.PHONY:		run
run:
		./corpmets

.PHONY:		mkenv
mkenv:
		conda env create -f enviornment.yml prefix=./env

.PHONY:		clean
clean:
		rm -fr $(MTARG)
		find . -type d -name __pycache__ -prune -exec rm -r {} \;

.PHONY:		cleanall
cleanall:	clean
		rm -fr data

