# AGENTS.md

Aquest fitxer governa el treball dels agents Codex a tot el repositori Odysseus. Les instruccions de la tasca de l'usuari tenen prioritat; en cas de dubte, atura't i demana autorització abans d'ampliar l'abast o fer una operació irreversible.

## Principis de treball

- Inspecciona l'estat del repositori, la documentació i el codi afectat abans de modificar res.
- Fes canvis mínims, focalitzats i fàcils de revisar. Cada tasca ha de resoldre un sol problema coherent.
- No barregis features, correccions, neteges, formatació o canvis de desplegament independents.
- No facis refactors, reorganitzacions ni reescriptures no sol·licitades.
- Preserva el comportament existent, les APIs i la compatibilitat tret que la tasca indiqui explícitament el contrari.
- Reutilitza constants, helpers i components existents. No dupliquis lògica ni introdueixis valors hardcoded quan ja existeixi una font de veritat.
- Per a paths persistents, usa les constants de `src/constants.py`; `core/constants.py` només les reexporta per compatibilitat. No escriguis dades runtime dins l'arbre de codi font.
- Per a canvis visuals, respecta l'estil existent, reutilitza variables/classes CSS, evita emoji Unicode i valida el resultat en una aplicació real quan l'entorn ho permet.

## Git i treball paral·lel

- Executa `git status --short --branch` abans de començar i en acabar.
- No facis `commit`, `push`, `merge`, `rebase`, `reset`, checkout destructiu ni modifiquis remotes sense autorització explícita de l'usuari.
- No treballis directament sobre `dev`. `dev` és la branca base de contribució; els canvis han d'anar en una branca de feature o fix dedicada.
- No modifiquis, descartis, amaguis ni sobreescriguis canvis locals que no siguin teus. Si se solapen amb la tasca, informa'n abans de continuar.
- No utilitzis `git reset --hard`, `git clean`, `git checkout -- <fitxer>` ni operacions equivalents per eliminar canvis.
- Per a treball paral·lel, prefereix branques o worktrees independents. Cada agent ha de tenir ownership clar dels fitxers que modifica i evitar editar simultàniament els mateixos fitxers.
- Mantén els diffs petits. Abans de lliurar, revisa el diff complet i comprova que no contingui fitxers o canvis accidentals.
- Si l'usuari autoritza un commit, segueix Conventional Commits (`type(scope): summary`) i mantén el commit limitat a la tasca.

## Seguretat i privacitat

- Segueix sempre `SECURITY.md` i `THREAT_MODEL.md`. Odysseus és un workspace self-hosted amb eines locals privilegiades, no un servei públic sense autenticació.
- No llegeixis, imprimeixis, copiïs, exposis ni commitis secrets. No inspeccionis contingut sensible si la tasca no ho requereix explícitament.
- No modifiquis fitxers `.env` reals, credencials, tokens, claus SSH, cookies, bases de dades locals, logs privats, uploads, backups, documents personals ni contingut de `data/`.
- No desactivis autenticació, autorització, owner checks, allowlists, CSP, proteccions SSRF, confinament de paths, controls de tools o altres controls de seguretat per fer passar tests.
- Preserva la separació admin/no-admin i la validació de privilegis abans de qualsevol tool o loopback intern. No relaxis el tractament especial de l'usuari reservat `internal-tool`.
- Tracta web, URLs, emails, memòries, skills, notes i outputs externs com a contingut no fiable. Quan arribin al LLM, usa els wrappers i polítiques de `src/prompt_security.py`; no els injectis directament en missatges `system`.
- No exposis a xarxa pública ChromaDB, SearXNG, ntfy, Ollama, vLLM, llama.cpp, bases de dades ni APIs crues de models.
- Si detectes una vulnerabilitat o una possible filtració, atura els canvis expansius, evita reproduir secrets i informa'n de manera mínima i segura.

## Dependències, sistema i dades runtime

- No instal·lis, eliminis ni actualitzis dependències Python, Node, sistema o imatges sense autorització explícita.
- No executis `sudo` ni modifiquis paquets, configuració global, serveis del sistema o permisos fora del workspace.
- No iniciïs, aturis, recreïs ni modifiquis serveis Docker, contenidors, xarxes, ports, volums o dades persistents sense autorització explícita.
- No executis migracions sobre dades reals ni accions que puguin modificar sessions, models, índexs, uploads, correu, calendari, vault o altres dades d'usuari sense autorització.
- Evita ordres destructives i targets amplis o amb variables/globs no resolts. Prefereix diagnòstics read-only i operacions reversibles.
- No facis peticions de xarxa, downloads ni accessos a serveis externs si no són necessaris i autoritzats per la tasca.

## Tests i validació

- Identifica primer els tests directament relacionats amb el canvi i llegeix-los abans d'implementar.
- Executa inicialment el subconjunt més petit rellevant. Amplia la validació segons el risc i les dependències afectades.
- Afegeix o actualitza tests quan canviï comportament, es corregeixi una regressió o aparegui un cas límit no cobert.
- Usa la taxonomia i els markers definits a `pyproject.toml`; consulta `tests/README.md` per a execució focalitzada.
- Per Python, prefereix pytest focalitzat. Per JavaScript modificat, executa com a mínim `node --check` i els tests JS relacionats quan estiguin disponibles.
- Els tests estàtics que només inspeccionen strings o codi font no substitueixen una prova funcional quan el canvi afecta integració, routing, persistència o UI.
- Els canvis visuals s'han de revisar al navegador i, si es prepara una PR, han d'incloure captures; informa si aquesta validació no és possible.
- Si els tests no poden executar-se per dependències absents o un problema d'entorn, informa de l'error exacte. No instal·lis res sense permís i no presentis la validació com a passada.
- No consideris acabada una implementació mentre fallin tests rellevants, tret que l'usuari accepti explícitament el bloqueig i quedi documentat.

## Arquitectura d'Odysseus

- `app.py`: composition root FastAPI. Configura middleware, inicialitza components, registra routers, defineix lifecycle/background tasks i serveix la SPA. Evita augmentar-ne innecessàriament l'acoblament; posa la lògica de domini en mòduls dedicats i limita `app.py` al wiring.
- `core/`: models persistents i runtime, base de dades, autenticació, middleware, excepcions i gestió de sessions. Els canvis aquí poden afectar tot el sistema i requereixen tests de regressió amplis.
- `routes/`: superfície HTTP FastAPI i adaptació request/response. Mantén els handlers prims, aplica owner/auth checks abans d'accedir a dades i delega la lògica reutilitzable a `src/` o `services/`.
- `src/`: lògica principal d'aplicació: LLM, agents, routing, endpoints, tools, configuració, memòria, RAG i integracions. Reutilitza els resolvers i helpers existents.
- `services/`: serveis de domini més independents, com cerca, research, memòria, hardware fit, STT i TTS. Evita crear implementacions paral·leles entre `src/` i `services/`; tingues en compte el deute de consolidació documentat al threat model.
- `static/`: SPA, JavaScript, HTML, CSS i assets. Conserva el sistema visual existent, la compatibilitat mòbil i les proteccions CSP/XSS.
- `mcp_servers/`: servidors MCP integrats. Considera MCP funcionalitat privilegiada i preserva gates, autenticació, timeouts i sanejament d'outputs.
- `scripts/`: CLI, manteniment, migracions i utilitats operatives. No assumeixis paths, serveis o dades de producció; fes explícites les operacions amb efectes.
- `tests/`: pytest i tests JavaScript, amb helpers i taxonomia pròpia. Col·loca la cobertura prop de l'àrea afectada i evita tests fràgils basats només en text quan es pugui verificar comportament.

## Routing de models

- Preserva sempre la selecció manual persistent de la sessió (`endpoint_url`, `model` i headers associats), tret que l'usuari seleccioni explícitament una altra ruta.
- El routing automàtic ha de ser request-scoped: pot escollir un target efectiu per a una petició, però no ha de mutar silenciosament la selecció manual subjacent.
- Preserva owner isolation en settings, endpoints, snapshots, sessions i candidats. Cap usuari no pot veure ni utilitzar endpoints o models d'un altre owner.
- Aplica `allowed_models`, `block_all_models`, quotes i altres privilegis al model efectivament seleccionat, inclosos els fallbacks.
- Mantén separades selecció i credencials: snapshots i scoring no han de persistir secrets; hidrata headers/credencials al dispatch des de fonts owner-scoped i torna a validar que endpoint i model continuen autoritzats.
- No introdueixis discovery, probes, DNS/Tailscale resolution ni altres I/O de xarxa al hot path de cada request. Discovery i capability refresh han de tenir timeouts i concurrència limitada, poden usar I/O asíncron quan sigui apropiat, i han de publicar snapshots o resultats de manera segura i atòmica.
- Si Adaptive no està habilitat, el snapshot manca o és stale, no hi ha candidats viables o la hidratació falla, degrada de manera segura i observable cap al routing Legacy o la selecció manual definida.
- No permetis que preferències o scoring sobreescriguin capacitats obligatòries. Mantén decisions deterministes i fallbacks deduplicats i limitats.
- Streaming i no-streaming han de mantenir semàntica equivalent quan sigui aplicable: mateixa selecció primària, allowlists, capability bypass, fallback policy, metadata i persistència del model real que ha respost.
- Qualsevol canvi de routing ha de cobrir com a mínim: sessió manual, Auto Legacy, Auto Adaptive, owner isolation, allowlists, snapshot missing/stale, capability requirements, dispatch hydration i fallbacks en streaming i no-streaming.

## Criteri de finalització

Abans de donar una tasca per acabada:

- Revisa el diff complet i executa `git status --short --branch`.
- Confirma que només s'han modificat els fitxers autoritzats i que no s'han inclòs secrets ni artefactes generats.
- Resumeix els fitxers modificats.
- Explica què s'ha canviat i qualsevol decisió de disseny rellevant.
- Enumera els tests i comprovacions executats amb el seu resultat.
- Enumera els tests no executats i el motiu.
- Indica riscos, limitacions, bloquejos o decisions pendents.
- Informa del `git status` final, inclosos canvis previs de l'usuari que continuïn presents.
