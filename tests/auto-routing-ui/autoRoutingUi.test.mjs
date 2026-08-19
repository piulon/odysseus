import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  autoPickerState,
  buildManualSelectionFields,
  buildPendingSessionFields,
  createLatestConfigController,
  createManualSelectionFormData,
  createPendingSessionFormData,
  createAutoSelectModelHandler,
  applyPendingModelPick,
  resolveDefaultPendingChat,
  replacePendingWithManualCandidate,
  createAutoPickerButton,
  loadAutoRoutingConfig,
  normalizeAutoCatalog,
  patchSessionAutoRoute,
  saveAutoTarget,
  validateAutoTarget,
} from '../../static/js/autoRoutingUi.js';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

const catalogPayload = {
  items: [
    {
      endpoint_id: 'owned-chat',
      endpoint_name: 'Owned chat',
      url: 'https://must-not-be-persisted.invalid/v1/chat/completions',
      models: ['chat-a'],
      models_extra: ['chat-b'],
    },
    {
      endpoint_id: 'owned-agent',
      endpoint_name: 'Owned agent',
      models: ['agent-a'],
      models_extra: [],
    },
  ],
};

function response(ok, json = {}) {
  return { ok, async json() { return json; } };
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.className = '';
    this.textContent = '';
    this.type = '';
  }
  appendChild(child) { this.children.push(child); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  click() { this.listeners.click(); }
}

test('Auto picker option is an accessible keyboard-selectable model row', () => {
  let selected = null;
  const documentFixture = { createElement: tag => new FakeElement(tag) };
  const row = createAutoPickerButton(documentFixture, true, value => { selected = value; });
  assert.equal(row.tagName, 'BUTTON');
  assert.equal(row.type, 'button');
  assert.match(row.className, /model-switch-item/);
  assert.equal(row.attributes['aria-pressed'], 'true');
  assert.equal(row.children[0].textContent, 'Auto');
  row.click();
  assert.equal(selected, false);
});

test('catalog keeps owner-scoped IDs/models and drops raw URLs', () => {
  const catalog = normalizeAutoCatalog(catalogPayload);
  assert.deepEqual(catalog, [
    { id: 'owned-chat', name: 'Owned chat', models: ['chat-a', 'chat-b'] },
    { id: 'owned-agent', name: 'Owned agent', models: ['agent-a'] },
  ]);
  assert.equal(JSON.stringify(catalog).includes('must-not-be-persisted'), false);
});

test('target validity distinguishes valid, unconfigured and stale', () => {
  const catalog = normalizeAutoCatalog(catalogPayload);
  assert.equal(validateAutoTarget(catalog, '', '').status, 'unconfigured');
  assert.equal(validateAutoTarget(catalog, 'owned-chat', '').status, 'unconfigured');
  assert.equal(validateAutoTarget(catalog, 'missing', 'chat-a').status, 'invalid');
  assert.equal(validateAutoTarget(catalog, 'owned-chat', 'missing').status, 'invalid');
  assert.equal(validateAutoTarget(catalog, 'owned-chat', 'chat-a').status, 'valid');
});

test('picker state is lane-specific and never replaces the manual model', () => {
  const catalog = normalizeAutoCatalog(catalogPayload);
  const prefs = {
    auto_chat_endpoint_id: 'owned-chat', auto_chat_model: 'chat-a',
    auto_agent_endpoint_id: '', auto_agent_model: '',
  };
  const session = { auto_route: true, model: 'manual-model', endpoint_url: 'manual-url' };
  const chat = autoPickerState(session, null, 'chat', prefs, catalog);
  const agent = autoPickerState(session, null, 'agent', prefs, catalog);
  assert.equal(chat.label, 'Auto');
  assert.equal(chat.warning, '');
  assert.equal(agent.label, 'Auto');
  assert.match(agent.warning, /not configured/i);
  assert.equal(session.model, 'manual-model');
  assert.equal(session.endpoint_url, 'manual-url');
});

test('loaded fork with auto_route false displays its manual model', () => {
  const state = autoPickerState(
    { auto_route: false, model: 'manual/fork-model' }, null, 'chat', {}, [],
  );
  assert.equal(state.label, 'fork-model');
  assert.equal(state.autoRoute, false);
});

test('existing-session Auto PATCH contains only auto_route and mutates after success', async () => {
  const calls = [];
  const session = { id: 's1', auto_route: false, model: 'manual', endpoint_url: 'manual-url' };
  const result = await patchSessionAutoRoute(session, true, async (url, init) => {
    calls.push([url, init]);
    return response(true, { auto_route: true });
  }, 'https://app.invalid');

  assert.equal(result.ok, true);
  assert.equal(session.auto_route, true);
  assert.equal(session.model, 'manual');
  assert.equal(session.endpoint_url, 'manual-url');
  assert.deepEqual([...calls[0][1].body.entries()], [['auto_route', 'true']]);
});

test('disabling Auto PATCHes only false and failure preserves authoritative local state', async () => {
  const sent = [];
  const session = { id: 's1', auto_route: true, model: 'manual', endpoint_url: 'manual-url' };
  const failed = await patchSessionAutoRoute(session, false, async (_url, init) => {
    sent.push([...init.body.entries()]);
    return response(false);
  }, '');
  assert.equal(failed.ok, false);
  assert.equal(session.auto_route, true);
  assert.deepEqual(sent, [[['auto_route', 'false']]]);
});

test('pending Auto can materialize without a manual target', () => {
  const fields = buildPendingSessionFields({ autoRoute: true });
  assert.deepEqual(fields, {
    endpoint_url: '', model: '', auto_route: 'true', skip_validation: 'true',
  });
  assert.equal(JSON.stringify(fields).includes('Auto"'), false);
});

test('pending manual and pending Auto with a real target preserve legacy route fields', () => {
  assert.deepEqual(buildPendingSessionFields({
    autoRoute: false, url: 'manual-url', modelId: 'manual-model', endpointId: 'ep-1',
  }), {
    endpoint_url: 'manual-url', model: 'manual-model', endpoint_id: 'ep-1',
    skip_validation: 'true',
  });
  assert.deepEqual(buildPendingSessionFields({
    autoRoute: true, url: 'manual-url', modelId: 'manual-model', endpointId: 'ep-1',
  }), {
    endpoint_url: 'manual-url', model: 'manual-model', endpoint_id: 'ep-1',
    auto_route: 'true', skip_validation: 'true',
  });
});

test('manual selection preserves its legacy target and explicitly disables Auto', () => {
  assert.deepEqual(buildManualSelectionFields({
    mid: 'manual-model', url: 'manual-url', endpointId: 'manual-endpoint',
  }), {
    model: 'manual-model', endpoint_url: 'manual-url',
    endpoint_id: 'manual-endpoint', auto_route: 'false',
  });
});

test('winner metadata cannot change persistent Auto picker state', () => {
  const session = { auto_route: true, model: 'manual-model' };
  const metadata = { requested_model: 'auto-primary', model: 'fallback-winner' };
  const state = autoPickerState(session, null, 'chat', {}, []);
  assert.equal(state.autoRoute, true);
  assert.equal(state.label, 'Auto');
  assert.equal(metadata.model, 'fallback-winner');
  assert.equal(session.model, 'manual-model');
});

test('config loading uses only owner-scoped prefs and models APIs', async () => {
  const urls = [];
  const fetchImpl = async (url) => {
    urls.push(url);
    if (url === '/api/prefs') return response(true, { auto_chat_model: 'chat-a' });
    if (url === '/api/models') return response(true, catalogPayload);
    throw new Error(`unexpected URL ${url}`);
  };
  const loaded = await loadAutoRoutingConfig(fetchImpl);
  assert.deepEqual(urls.sort(), ['/api/models', '/api/prefs']);
  assert.equal(loaded.prefs.auto_chat_model, 'chat-a');
  assert.equal(loaded.catalog[0].id, 'owned-chat');
});

test('chat and agent targets save independent endpoint/model prefs', async () => {
  const calls = [];
  const fetchImpl = async (url, init = {}) => {
    calls.push([url, init]);
    if (init.method === 'PUT') return response(true);
    return response(true, {});
  };
  const catalog = normalizeAutoCatalog(catalogPayload);
  assert.equal((await saveAutoTarget('chat', 'owned-chat', 'chat-a', catalog, fetchImpl)).ok, true);
  assert.equal((await saveAutoTarget('agent', 'owned-agent', 'agent-a', catalog, fetchImpl)).ok, true);
  assert.deepEqual(calls.map(([url]) => url), [
    '/api/prefs/auto_chat_endpoint_id', '/api/prefs/auto_chat_model',
    '/api/prefs/auto_agent_endpoint_id', '/api/prefs/auto_agent_model',
  ]);
  assert.equal(calls.some(([url]) => url.includes('model-endpoints') || url.includes('auth/settings')), false);
});

test('invalid or stale pair is never persisted or silently substituted', async () => {
  let calls = 0;
  const result = await saveAutoTarget(
    'chat', 'owned-chat', 'missing', normalizeAutoCatalog(catalogPayload),
    async () => { calls += 1; return response(true); },
  );
  assert.equal(result.ok, false);
  assert.equal(result.status, 'invalid');
  assert.equal(calls, 0);
});

test('clearing a lane persists two empty owner-scoped preferences', async () => {
  const calls = [];
  const result = await saveAutoTarget('agent', '', '', [], async (url, init) => {
    calls.push([url, JSON.parse(init.body)]);
    return response(true);
  });
  assert.equal(result.ok, true);
  assert.deepEqual(calls, [
    ['/api/prefs/auto_agent_endpoint_id', { value: '' }],
    ['/api/prefs/auto_agent_model', { value: '' }],
  ]);
});

test('partial prefs failure reloads authoritative prefs and reports failure', async () => {
  const urls = [];
  const fetchImpl = async (url, init = {}) => {
    urls.push(url);
    if (url.endsWith('_endpoint_id')) return response(true);
    if (url.endsWith('_model')) return response(false);
    if (url === '/api/prefs') return response(true, {
      auto_chat_endpoint_id: 'owned-chat', auto_chat_model: '',
    });
    if (url === '/api/models') return response(true, catalogPayload);
    throw new Error(`unexpected URL ${url}`);
  };
  const result = await saveAutoTarget(
    'chat', 'owned-chat', 'chat-a', normalizeAutoCatalog(catalogPayload), fetchImpl,
  );
  assert.equal(result.ok, false);
  assert.equal(result.authoritative.prefs.auto_chat_model, '');
  assert.deepEqual(urls.slice(-2).sort(), ['/api/models', '/api/prefs']);
});

test('settings-style refresh ignores an older response after a newer refresh', async () => {
  const requestA = deferred();
  const requestB = deferred();
  const pending = [requestA, requestB];
  const applied = [];
  const controller = createLatestConfigController(
    () => pending.shift().promise,
    value => applied.push(value),
  );

  const refreshA = controller.refresh();
  const refreshB = controller.refresh();
  requestB.resolve({ version: 'new' });
  assert.equal(await refreshB, true);
  requestA.resolve({ version: 'old' });
  assert.equal(await refreshA, false);
  assert.deepEqual(applied, [{ version: 'new' }]);
});

test('picker-style config event invalidates an older refresh response', async () => {
  const request = deferred();
  const applied = [];
  const controller = createLatestConfigController(
    () => request.promise,
    value => applied.push(value),
  );

  const refresh = controller.refresh();
  controller.applyLatest({ version: 'event' });
  request.resolve({ version: 'old-refresh' });
  assert.equal(await refresh, false);
  assert.deepEqual(applied, [{ version: 'event' }]);
});

test('an older settings save result cannot repaint a newer user save', () => {
  const applied = [];
  const controller = createLatestConfigController(async () => ({}), value => applied.push(value));
  const saveA = controller.begin();
  const saveB = controller.begin();
  assert.equal(controller.applyIfCurrent(saveB, { version: 'new-save' }), true);
  assert.equal(controller.applyIfCurrent(saveA, { version: 'old-save' }), false);
  assert.deepEqual(applied, [{ version: 'new-save' }]);
});

test('config controller setup is idempotent across init and reopen', () => {
  let listenerRegistrations = 0;
  const controller = createLatestConfigController(async () => ({}), () => {});
  assert.equal(controller.initialize(() => { listenerRegistrations += 1; }), true);
  assert.equal(controller.initialize(() => { listenerRegistrations += 1; }), false);
  assert.equal(listenerRegistrations, 1);
});

test('pending Auto materialization FormData has routing flags and no fictitious target', () => {
  const body = createPendingSessionFormData({ autoRoute: true }, 'Auto pending');
  assert.deepEqual([...body.entries()], [
    ['name', 'Auto pending'],
    ['endpoint_url', ''],
    ['model', ''],
    ['auto_route', 'true'],
    ['skip_validation', 'true'],
  ]);
});

test('manual picker FormData preserves the legacy target and disables Auto', () => {
  const body = createManualSelectionFormData({
    mid: 'manual-model', url: 'manual-url', endpointId: 'manual-endpoint',
  });
  assert.deepEqual([...body.entries()], [
    ['model', 'manual-model'],
    ['endpoint_url', 'manual-url'],
    ['endpoint_id', 'manual-endpoint'],
    ['auto_route', 'false'],
  ]);
});

test('a default arriving after catalog refresh cannot replace an active Auto pending', () => {
  let pending = { autoRoute: true, source: 'auto' };
  const applied = replacePendingWithManualCandidate(
    () => pending,
    value => { pending = value; },
    {
      url: 'http://local.invalid/v1', modelId: 'qwen3-vl:8b',
      endpointId: 'ollama', source: 'default',
    },
  );

  assert.equal(applied, false);
  assert.deepEqual(pending, { autoRoute: true, source: 'auto' });
  assert.equal(autoPickerState(null, pending, 'chat', {}, []).label, 'Auto');
});

test('a later manual fallback cannot clear Auto or become the active picker selection', () => {
  let pending = {
    autoRoute: true, source: 'auto',
    url: 'manual-fallback-url', modelId: 'manual-fallback', endpointId: 'manual-ep',
  };
  const original = { ...pending };
  const applied = replacePendingWithManualCandidate(
    () => pending,
    value => { pending = value; },
    {
      url: 'new-default-url', modelId: 'new-default',
      endpointId: 'new-ep', source: 'fallback',
    },
  );

  assert.equal(applied, false);
  assert.deepEqual(pending, original);
  assert.equal(pending.autoRoute, true);
  assert.equal(autoPickerState(null, pending, 'chat', {}, []).label, 'Auto');
});

test('a first default still initializes an empty manual pending when Auto is off', () => {
  let pending = null;
  const applied = replacePendingWithManualCandidate(
    () => pending,
    value => { pending = value; },
    {
      url: 'manual-url', modelId: 'manual-default',
      endpointId: 'manual-ep', source: 'default',
    },
  );

  assert.equal(applied, true);
  assert.deepEqual(pending, {
    url: 'manual-url', modelId: 'manual-default', endpointId: 'manual-ep',
    source: 'default', autoRoute: false,
  });
  assert.equal(autoPickerState(null, pending, 'chat', {}, []).label, 'manual-default');
});

test('real endpoint-added event cannot let a refreshed automatic pick replace Auto', async () => {
  const refresh = deferred();
  let pending = null;
  let pickerLabel = 'Select model';
  const target = new EventTarget();
  const handler = createAutoSelectModelHandler({
    getState: () => ({ current: null, pending }),
    refreshModels: () => refresh.promise,
    resolveModel: detail => ({
      mid: detail.modelId, url: detail.url, endpointId: detail.endpointId,
    }),
    pick: model => {
      applyPendingModelPick(
        () => pending,
        value => { pending = value; },
        model,
        { automatic: true },
      );
      pickerLabel = autoPickerState(null, pending, 'chat', {}, []).label;
    },
  });
  target.addEventListener('odysseus:auto-select-model', handler);

  const event = new Event('odysseus:auto-select-model');
  event.detail = {
    endpointId: 'ollama', modelId: 'qwen3-vl:8b',
    url: 'http://local.invalid/v1',
  };
  target.dispatchEvent(event);
  pending = { autoRoute: true, source: 'auto' };
  pickerLabel = 'Auto';
  refresh.resolve();
  await new Promise(resolve => setImmediate(resolve));

  assert.deepEqual(pending, { autoRoute: true, source: 'auto' });
  assert.equal(pickerLabel, 'Auto');
});

test('automatic model pick is rejected at the pending write boundary', () => {
  let pending = { autoRoute: true, source: 'auto' };
  const applied = applyPendingModelPick(
    () => pending,
    value => { pending = value; },
    { mid: 'qwen3-vl:8b', url: 'manual-url', endpointId: 'ollama' },
    { automatic: true },
  );
  assert.equal(applied, false);
  assert.deepEqual(pending, { autoRoute: true, source: 'auto' });
});

test('an explicit manual pick still disables Auto at the same write boundary', () => {
  let pending = { autoRoute: true, source: 'auto' };
  const applied = applyPendingModelPick(
    () => pending,
    value => { pending = value; },
    { mid: 'manual-model', url: 'manual-url', endpointId: 'manual-ep' },
  );
  assert.equal(applied, true);
  assert.deepEqual(pending, {
    url: 'manual-url', modelId: 'manual-model', endpointId: 'manual-ep',
    source: 'manual', autoRoute: false,
  });
});

test('real default resolver cannot overwrite Auto selected during catalog refresh', async () => {
  const catalogRefresh = deferred();
  let pending = null;
  const resolution = resolveDefaultPendingChat({
    getPending: () => pending,
    setPending: value => { pending = value; },
    ensureCatalog: () => catalogRefresh.promise,
    loadDefault: async () => ({
      endpoint_url: 'http://100.87.190.46:11434/v1/chat/completions',
      model: 'qwen3-vl:8b',
      endpoint_id: '1ba63d8',
    }),
    modelExists: () => true,
    firstFallback: () => null,
  });

  pending = { autoRoute: true, source: 'auto' };
  catalogRefresh.resolve();
  assert.equal(await resolution, false);
  assert.deepEqual(pending, { autoRoute: true, source: 'auto' });
  assert.equal(autoPickerState(null, pending, 'chat', {}, []).label, 'Auto');
});
