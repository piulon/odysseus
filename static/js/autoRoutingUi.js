const AUTO_PREF_KEYS = Object.freeze({
  chat: Object.freeze({
    endpoint: 'auto_chat_endpoint_id',
    model: 'auto_chat_model',
  }),
  agent: Object.freeze({
    endpoint: 'auto_agent_endpoint_id',
    model: 'auto_agent_model',
  }),
});

function _clean(value) {
  return String(value || '').trim();
}

function _lane(lane) {
  return lane === 'agent' ? 'agent' : 'chat';
}

export function normalizeAutoCatalog(payload) {
  const items = payload && Array.isArray(payload.items) ? payload.items : [];
  return items.flatMap(item => {
    const id = _clean(item && item.endpoint_id);
    if (!id) return [];
    const models = [...(Array.isArray(item.models) ? item.models : []),
      ...(Array.isArray(item.models_extra) ? item.models_extra : [])]
      .map(_clean)
      .filter(Boolean);
    return [{
      id,
      name: _clean(item.endpoint_name) || id,
      models: [...new Set(models)],
    }];
  });
}

export function validateAutoTarget(catalog, endpointId, model) {
  const cleanEndpoint = _clean(endpointId);
  const cleanModel = _clean(model);
  if (!cleanEndpoint || !cleanModel) {
    return { status: 'unconfigured', endpoint: null };
  }
  const endpoint = (catalog || []).find(item => item.id === cleanEndpoint) || null;
  if (!endpoint || !endpoint.models.includes(cleanModel)) {
    return { status: 'invalid', endpoint };
  }
  return { status: 'valid', endpoint };
}

export function autoTargetForLane(lane, prefs = {}, catalog = []) {
  const keys = AUTO_PREF_KEYS[_lane(lane)];
  const endpointId = _clean(prefs[keys.endpoint]);
  const model = _clean(prefs[keys.model]);
  return {
    endpointId,
    model,
    ...validateAutoTarget(catalog, endpointId, model),
  };
}

export function autoPickerState(session, pending, lane, prefs = {}, catalog = []) {
  const autoRoute = !!(
    (session && session.auto_route)
    || (!session && pending && pending.autoRoute)
  );
  const manualModel = _clean(
    session ? session.model : (pending && pending.modelId),
  );
  const target = autoTargetForLane(lane, prefs, catalog);
  return {
    autoRoute,
    label: autoRoute ? 'Auto' : (manualModel ? manualModel.split('/').pop() : 'Select model'),
    warning: autoRoute && target.status !== 'valid'
      ? 'Auto routing is not configured'
      : '',
    targetStatus: target.status,
  };
}

export function createAutoPickerButton(documentImpl, autoRoute, onToggle) {
  const row = documentImpl.createElement('button');
  row.type = 'button';
  row.className = 'model-switch-item model-switch-auto';
  row.setAttribute('aria-pressed', autoRoute ? 'true' : 'false');

  const name = documentImpl.createElement('span');
  name.className = 'mp-model-name';
  name.textContent = 'Auto';
  row.appendChild(name);

  const description = documentImpl.createElement('span');
  description.className = 'model-switch-ep';
  description.textContent = autoRoute ? 'On · switch to manual' : 'Automatic routing';
  row.appendChild(description);
  row.addEventListener('click', () => onToggle(!autoRoute));
  return row;
}

export async function patchSessionAutoRoute(
  session,
  enabled,
  fetchImpl = fetch,
  apiBase = window.location.origin,
) {
  const body = new FormData();
  body.append('auto_route', enabled ? 'true' : 'false');
  try {
    const result = await fetchImpl(
      `${apiBase}/api/session/${encodeURIComponent(session.id)}`,
      { method: 'PATCH', body },
    );
    if (!result.ok) return { ok: false };
    session.auto_route = !!enabled;
    return { ok: true };
  } catch (_) {
    return { ok: false };
  }
}

export function buildPendingSessionFields(pending = {}) {
  const fields = {
    endpoint_url: _clean(pending.url),
    model: _clean(pending.modelId),
  };
  if (_clean(pending.endpointId)) fields.endpoint_id = _clean(pending.endpointId);
  if (pending.autoRoute) fields.auto_route = 'true';
  if (pending.autoRoute || (fields.endpoint_url && fields.model)) {
    fields.skip_validation = 'true';
  }
  return fields;
}

export function buildManualSelectionFields(model = {}) {
  const fields = {
    model: _clean(model.mid),
    endpoint_url: _clean(model.url),
  };
  if (_clean(model.endpointId)) fields.endpoint_id = _clean(model.endpointId);
  fields.auto_route = 'false';
  return fields;
}

function _formDataFromFields(fields) {
  const body = new FormData();
  Object.entries(fields).forEach(([key, value]) => body.append(key, value));
  return body;
}

export function createPendingSessionFormData(pending = {}, name = '') {
  return _formDataFromFields({ name, ...buildPendingSessionFields(pending) });
}

export function createManualSelectionFormData(model = {}) {
  return _formDataFromFields(buildManualSelectionFields(model));
}

export function replacePendingWithManualCandidate(getPending, setPending, candidate) {
  const current = getPending();
  if (current && (current.autoRoute || current.source === 'manual')) return false;
  setPending({ ...candidate, autoRoute: false });
  return true;
}

function _blocksAutomaticPending(current) {
  return !!(current && (current.autoRoute || (current.modelId && current.source === 'manual')));
}

export async function resolveDefaultPendingChat({
  getPending,
  setPending,
  ensureCatalog,
  loadDefault,
  modelExists,
  firstFallback,
  onDefaultLoaded = () => {},
}) {
  if (_blocksAutomaticPending(getPending())) return false;
  await ensureCatalog();
  if (_blocksAutomaticPending(getPending())) return false;

  let defaultChat = null;
  try { defaultChat = await loadDefault(); } catch (_) {}
  if (_blocksAutomaticPending(getPending())) return false;
  if (defaultChat && defaultChat.endpoint_url && defaultChat.model
    && modelExists(defaultChat.model, defaultChat.endpoint_url)) {
    onDefaultLoaded(defaultChat);
    return replacePendingWithManualCandidate(getPending, setPending, {
      url: defaultChat.endpoint_url,
      modelId: defaultChat.model,
      endpointId: defaultChat.endpoint_id || '',
      source: 'default',
    });
  }

  if (_blocksAutomaticPending(getPending())) return false;
  const fallback = firstFallback();
  if (!fallback) return false;
  return replacePendingWithManualCandidate(
    getPending,
    setPending,
    { ...fallback, source: 'fallback' },
  );
}

export function applyPendingModelPick(
  getPending,
  setPending,
  model,
  { automatic = false } = {},
) {
  const current = getPending();
  if (automatic && current && current.autoRoute) return false;
  setPending({
    url: model.url,
    modelId: model.mid,
    endpointId: model.endpointId,
    source: 'manual',
    autoRoute: false,
  });
  return true;
}

function _hasSelectedRoute({ current, pending } = {}) {
  return !!(
    (current && (current.model || current.auto_route))
    || (pending && (pending.modelId || pending.autoRoute))
  );
}

export function createAutoSelectModelHandler({
  getState,
  refreshModels,
  resolveModel,
  pick,
}) {
  return async event => {
    if (_hasSelectedRoute(getState())) return false;
    try { await refreshModels(); } catch (_) {}
    if (_hasSelectedRoute(getState())) return false;
    const model = resolveModel((event && event.detail) || {});
    if (!model) return false;
    await pick(model, { automatic: true });
    return true;
  };
}

export function createLatestConfigController(load, apply) {
  let generation = 0;
  let initialized = false;
  return {
    begin() {
      generation += 1;
      return generation;
    },
    isCurrent(token) {
      return token === generation;
    },
    applyIfCurrent(token, value) {
      if (token !== generation) return false;
      apply(value);
      return true;
    },
    async refresh() {
      const requestGeneration = this.begin();
      try {
        const value = await load();
        return this.applyIfCurrent(requestGeneration, value);
      } catch (error) {
        if (requestGeneration !== generation) return false;
        throw error;
      }
    },
    applyLatest(value) {
      generation += 1;
      apply(value);
    },
    invalidate() {
      return this.begin();
    },
    initialize(register) {
      if (initialized) return false;
      initialized = true;
      register();
      return true;
    },
  };
}

export async function loadAutoRoutingConfig(fetchImpl = fetch) {
  const [prefsResponse, modelsResponse] = await Promise.all([
    fetchImpl('/api/prefs', { credentials: 'same-origin' }),
    fetchImpl('/api/models', { credentials: 'same-origin' }),
  ]);
  if (!prefsResponse.ok || !modelsResponse.ok) {
    throw new Error('Unable to load automatic routing settings');
  }
  const [prefs, models] = await Promise.all([
    prefsResponse.json(), modelsResponse.json(),
  ]);
  return {
    prefs: prefs && typeof prefs === 'object' ? prefs : {},
    catalog: normalizeAutoCatalog(models),
  };
}

async function _putPref(fetchImpl, key, value) {
  return fetchImpl(`/api/prefs/${encodeURIComponent(key)}`, {
    method: 'PUT',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  });
}

export async function saveAutoTarget(
  lane,
  endpointId,
  model,
  catalog,
  fetchImpl = fetch,
) {
  const normalizedLane = _lane(lane);
  const validation = validateAutoTarget(catalog, endpointId, model);
  const clearing = !_clean(endpointId) && !_clean(model);
  if (!clearing && validation.status !== 'valid') {
    return { ok: false, status: validation.status };
  }

  const keys = AUTO_PREF_KEYS[normalizedLane];
  try {
    const endpointResponse = await _putPref(fetchImpl, keys.endpoint, _clean(endpointId));
    if (!endpointResponse.ok) throw new Error('endpoint preference failed');
    const modelResponse = await _putPref(fetchImpl, keys.model, _clean(model));
    if (!modelResponse.ok) throw new Error('model preference failed');
    return { ok: true, status: clearing ? 'unconfigured' : 'valid' };
  } catch (_) {
    let authoritative = { prefs: {}, catalog: catalog || [] };
    try { authoritative = await loadAutoRoutingConfig(fetchImpl); } catch (_) {}
    return { ok: false, status: 'save_failed', authoritative };
  }
}

export { AUTO_PREF_KEYS };
