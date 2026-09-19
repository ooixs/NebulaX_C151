"""Contract tests run without installing inference dependencies."""
import csv
import hashlib
import io
import json
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app import server


class Frame:
    def __init__(self, rows):
        self.rows = rows

    def to_json(self, **kwargs):
        return json.dumps(self.rows)

    def __getitem__(self, columns):
        return self

    def to_csv(self, **kwargs):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(self.rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        return buffer.getvalue()


class ContractTests(unittest.TestCase):
    def setUp(self):
        server.RUNS.clear()
        diagnostic_patch = patch.object(server.diagnostics, 'build', return_value={'version':1,'items':[]})
        diagnostic_patch.start()
        self.addCleanup(diagnostic_patch.stop)

    def request(self, path, body, content_type='application/json', origin=None, host='127.0.0.1:8765'):
        handler = object.__new__(server.Handler)
        handler.path = path
        handler.headers = {'Content-Length':str(len(body)), 'Content-Type':content_type, 'Host':host}
        if origin:
            handler.headers['Origin'] = origin
        handler.rfile = io.BytesIO(body)
        replies = []
        handler.send = lambda status, data, *args, **kwargs: replies.append((status, data))
        handler.do_POST()
        return replies[0]

    def test_multipart_upload_to_prediction_to_export(self):
        boundary = 'nebulax-test-boundary'
        body = ('--'+boundary+'\r\nContent-Disposition: form-data; name="system"\r\n\r\nrail\r\n'
                '--'+boundary+'\r\nContent-Disposition: form-data; name="files"; filename="Test1.csv"\r\n'
                'Content-Type: text/csv\r\n\r\n1,2,3\r\n--'+boundary+'--\r\n').encode()
        stub = types.ModuleType('common.inference')
        stub.predict = lambda *_: Frame([{'file_id':'Test1.csv','prediction':'Normal'}])
        with patch.dict('sys.modules', {'common.inference':stub}), patch.object(server, 'model_status', return_value={'ready':True,'sha256':'verified'}):
            status, run = self.request('/api/analyze', body, 'multipart/form-data; boundary='+boundary)
        self.assertEqual(status, 200)
        self.assertEqual(run['files'][0]['name'], 'Test1.csv')
        status, archive = self.request('/api/export', json.dumps({'ids':[run['id']]}).encode())
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            self.assertEqual(zipped.read('rail_predictions.csv'), b'file_id,prediction\nTest1.csv,Normal\n')

    def test_cross_origin_and_empty_requests_rejected(self):
        self.assertEqual(self.request('/api/export', b'{}', origin='https://unrelated.example')[0], 403)
        self.assertEqual(self.request('/api/analyze', b'')[0], 413)

    def test_https_same_origin_is_accepted(self):
        self.assertNotEqual(
            self.request('/api/export', b'{}', origin='https://127.0.0.1:8765')[0],
            403,
        )

    def test_loopback_proxy_origin_is_accepted(self):
        self.assertNotEqual(
            self.request(
                '/api/export',
                b'{}',
                origin='http://127.0.0.1:18080',
                host='nebulax-control-room.example.run.app',
            )[0],
            403,
        )

    def test_upload_boundaries(self):
        for key, files in [
            ('rail', []), ('invalid', [('x.csv', b'1')]),
            ('rail', [('../x.csv', b'1')]), ('rail', [('a\\b.csv', b'1')]),
            ('rail', [('=x.csv', b'1')]), ('rail', [('x.csv', b'')]),
            ('acv', [('x.csv', b'1')]),
            ('rail', [('x.csv', b'1'), ('X.csv', b'2')]),
            ('door', [('x.csv', b'1'), ('y.csv', b'2')]),
        ]:
            with self.subTest(key=key, files=files), self.assertRaises(ValueError):
                server.validate_files(key, files)
        server.validate_files('rail', [('Test1.csv', b'1,2')])

    def test_invalid_outputs_cannot_be_exported(self):
        examples = [('shm', [{'file_id':'a.csv','prediction':float('nan')}]),
                    ('shm', [{'file_id':'a.csv','prediction':-1}]),
                    ('rail', [{'file_id':'a.csv','prediction':'Side III'}]),
                    ('acv', [{'file_id':'a.xlsx','ranked_cars':'01|01'}]),
                    ('door', [{'start_time':'2023-7-5-0-0-3-760','end_time':'2023-7-5-0-0-0-0','prediction':'Normal'}])]
        for key, rows in examples:
            with self.subTest(key=key), self.assertRaises(ValueError):
                server.validate_rows(key, rows)
        with self.assertRaises(ValueError):
            server.validate_rows('rail', [{'file_id':'wrong.csv','prediction':'Normal'}], ['expected.csv'])

    def test_manifest_integrity_and_selection(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(server, 'ROOT', Path(temp)):
            directory = Path(temp) / 'Door/model'
            directory.mkdir(parents=True)
            self.assertFalse(server.model_status('door')['ready'])
            artifact = directory / 'door_model.joblib'
            artifact.write_bytes(b'trusted model placeholder')
            manifest = directory / 'active_model.json'
            record = dict(schema_version=1, file=artifact.name, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), run_id='research')
            manifest.write_text(json.dumps(record))
            self.assertTrue(server.model_status('door')['ready'])
            artifact.write_bytes(b'tampered')
            self.assertFalse(server.model_status('door')['ready'])
            manifest.write_text('[]')
            self.assertFalse(server.model_status('door')['ready'])

    def test_model_details_follow_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(server, 'ROOT', Path(temp)):
            directory = Path(temp) / 'Rail Corrugation/model'
            directory.mkdir(parents=True)
            artifact = directory / 'rail_model.joblib'
            artifact.write_bytes(b'first checkpoint')
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            manifest = directory / 'active_model.json'
            record = dict(schema_version=1, file=artifact.name, sha256=digest, run_id='selected')
            manifest.write_text(json.dumps(record))
            with patch.object(server, 'MODEL_DETAILS', {digest: {'name': 'Selected ensemble'}}):
                self.assertEqual(server.model_status('rail')['details']['name'], 'Selected ensemble')
                artifact.write_bytes(b'replacement checkpoint')
                self.assertFalse(server.model_status('rail')['ready'])
                record['sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                manifest.write_text(json.dumps(record))
                self.assertIsNone(server.model_status('rail')['details'])

    def test_live_routes_shared_predictor_and_preserves_csv(self):
        rows = [{'file_id':'Test1.csv','prediction':'Side I'}]
        calls = []
        def predict(name, path):
            calls.append(name)
            self.assertEqual((path / 'Test1.csv').read_bytes(), b'sensor data')
            return Frame(rows)
        stub = types.ModuleType('common.inference')
        stub.predict = predict
        model = dict(ready=True, sha256='abc', artifact='rail_model.joblib', run_id='research')
        with patch.dict('sys.modules', {'common.inference':stub}), patch.object(server, 'model_status', return_value=model):
            run = server.analyze('rail', [('Test1.csv', b'sensor data')])
        self.assertEqual(calls, ['Rail Corrugation'])
        self.assertEqual(run['csv'], b'file_id,prediction\nTest1.csv,Side I\n')
        self.assertFalse(run['preview'])
        self.assertEqual(run['files'][0]['sha256'], hashlib.sha256(b'sensor data').hexdigest())
        self.assertNotIn('csv', server.public_run(run))

    def test_no_result_if_model_changes_during_run(self):
        stub = types.ModuleType('common.inference')
        stub.predict = lambda *_: Frame([{'file_id':'a.csv','prediction':'Normal'}])
        with patch.dict('sys.modules', {'common.inference':stub}), patch.object(server, 'model_status', side_effect=[dict(ready=True, sha256='a'), dict(ready=True, sha256='b')]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                server.analyze('rail', [('a.csv', b'1')])
        self.assertEqual(server.RUNS, {})

    def test_submission_archive_exact_names_no_preview(self):
        ids = []
        for key in server.SYSTEMS:
            rows = ([{'start_time':'2023-7-5-0-0-0-0','end_time':'2023-7-5-0-0-3-760','prediction':'Normal'}] if key=='door' else [{'file_id':'x.xlsx','ranked_cars':'01|02'}] if key=='acv' else [{'file_id':'x.csv','prediction':'0.12345678901234567' if key=='shm' else 'Side I'}])
            run = server.save_run(key, rows, [], {'sha256':'model'}, 0)
            run['csv'] = server.csv_bytes(key, rows)
            ids.append(run['id'])
        with zipfile.ZipFile(io.BytesIO(server.export_zip(ids))) as archive:
            self.assertEqual(set(archive.namelist()), {s['output'] for s in server.SYSTEMS.values()})
            self.assertTrue(all('/' not in name for name in archive.namelist()))
            self.assertEqual(archive.read('rail_predictions.csv'), b'file_id,prediction\nx.csv,Side I\n')
        preview = server.save_run('rail', [], [], None, 0, True)
        with self.assertRaisesRegex(ValueError, 'live'):
            server.export_zip([preview['id']])
        with self.assertRaisesRegex(ValueError, 'once'):
            server.export_zip([ids[0], ids[0]])

    def test_multiple_batches_merge_and_newest_duplicate_wins(self):
        old = server.save_run('rail', [], [], {'sha256':'same'}, 0)
        old['csv'] = b'file_id,prediction\na.csv,Normal\nb.csv,Side I\n'
        new = server.save_run('rail', [], [], {'sha256':'same'}, 0)
        new['csv'] = b'file_id,prediction\na.csv,Side II\n'
        new['created'] = old['created']
        for ids in ([old['id'], new['id']], [new['id'], old['id']]):
            with zipfile.ZipFile(io.BytesIO(server.export_zip(ids))) as archive:
                rows = list(csv.DictReader(io.StringIO(archive.read('rail_predictions.csv').decode())))
                self.assertEqual({r['file_id']:r['prediction'] for r in rows}, {'a.csv':'Side II','b.csv':'Side I'})
        new['model']['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different models'):
            server.export_zip([old['id'],new['id']])

    def test_all_subsystems_route_through_shared_interface(self):
        cases = {
            'door': ('Test.csv', [{'start_time':'2023-7-5-0-0-0-0','end_time':'2023-7-5-0-0-3-760','prediction':'Normal'}]),
            'acv': ('case.xlsx', [{'file_id':'case.xlsx','ranked_cars':'03|01|02'}]),
            'rail': ('test.csv', [{'file_id':'test.csv','prediction':'Side II'}]),
            'shm': ('test.csv', [{'file_id':'test.csv','prediction':0.12345678901234567}]),
        }
        for key, (filename, rows) in cases.items():
            with self.subTest(key=key):
                calls = []
                def predict(name, path):
                    calls.append((name, path.is_file()))
                    return Frame(rows)
                stub = types.ModuleType('common.inference')
                stub.predict = predict
                with patch.dict('sys.modules', {'common.inference':stub}), patch.object(server, 'model_status', return_value={'ready':True, 'sha256':'abc'}):
                    run = server.analyze(key, [(filename, b'data')])
                self.assertEqual(calls, [(server.SYSTEMS[key]['name'], key=='door')])
                self.assertEqual(run['csv'], Frame(rows).to_csv().encode())

    def test_operator_downloads_keep_submission_schema_separate(self):
        run = server.save_run('shm', [{'file_id':'x.csv','prediction':0.123}], [], {'sha256':'one'}, 0,
                              csv_content=b'file_id,prediction\nx.csv,0.12345678901234567\n')
        friendly = server.operator_csv(run).decode()
        self.assertEqual(friendly, 'Source file,Estimated fatigue damage\nx.csv,0.12345678901234567\n')
        with zipfile.ZipFile(io.BytesIO(server.export_zip([run['id']]))) as zipped:
            self.assertEqual(zipped.read('shm_predictions.csv'), run['csv'])

    def test_selected_download_contains_only_checked_files(self):
        rows = [{'file_id':'one.csv','prediction':'Normal'}, {'file_id':'two.csv','prediction':'Side II'}]
        run = server.save_run('rail', rows, [], {'sha256':'same'}, 0, csv_content=server.csv_bytes('rail',rows))
        data = server.selected_zip([{'id':run['id'],'file':'two.csv'}])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            self.assertEqual(archive.namelist(), ['rail_check_results.csv'])
            self.assertEqual(archive.read('rail_check_results.csv').decode(), 'Source file,Rail corrugation result\ntwo.csv,Side II\n')
        with self.assertRaises(ValueError):
            server.selected_zip([{'id':run['id'],'file':'missing.csv'}])
        with self.assertRaises(ValueError):
            server.selected_zip([])

    def test_task_download_groups_multiple_source_files(self):
        rows = [{'file_id':'one.csv','prediction':'Normal'}, {'file_id':'two.csv','prediction':'Side II'}]
        rail = server.save_run('rail', rows, [], {'sha256':'same'}, 0, csv_content=server.csv_bytes('rail',rows))
        structural = [{'file_id':'stress.csv','prediction':'0.12345678901234567'}]
        shm = server.save_run('shm', structural, [], {'sha256':'other'}, 0, csv_content=server.csv_bytes('shm',structural))
        selection = [{'id':rail['id'],'file':r['file_id']} for r in rows]+[{'id':shm['id'],'file':'stress.csv'}]
        with zipfile.ZipFile(io.BytesIO(server.selected_zip(selection))) as archive:
            self.assertEqual(set(archive.namelist()), {'rail_condition/one_results.csv','rail_condition/two_results.csv','shm_check_results.csv'})
            self.assertIn(b'two.csv,Side II', archive.read('rail_condition/two_results.csv'))
            self.assertIn(b'0.12345678901234567', archive.read('shm_check_results.csv'))

    def test_door_export_uses_recording_clock_to_seconds(self):
        self.assertEqual(server.readable_time('2023-7-5-0-11-17-664'), '05 Jul 2023, 00:11:17')
        run = dict(system='door', rows=[{'start_time':'2023-7-5-0-0-0-0','end_time':'2023-7-5-0-0-3-760','prediction':'Normal'}])
        self.assertEqual(server.operator_rows(run)[0]['Movement end (recording time)'], '05 Jul 2023, 00:00:03')

    def test_history_survives_restart_and_more_than_forty_checks(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(server,'HISTORY_PATH',None):
            path = Path(folder) / 'results.sqlite3'
            server.configure_history(path)
            for i in range(70):
                server.save_run('rail', [{'file_id':f'{i}.csv','prediction':'Normal'}], [], {'sha256':'one'}, 0,
                                csv_content=f'file_id,prediction\n{i}.csv,Normal\n'.encode())
            ids = list(server.RUNS)
            server.RUNS.clear()
            server.configure_history(path)
            self.assertEqual(list(server.RUNS), ids)
            with zipfile.ZipFile(io.BytesIO(server.export_zip(ids))) as zipped:
                self.assertEqual(len(list(csv.DictReader(io.StringIO(zipped.read('rail_predictions.csv').decode())))),70)

    def test_queued_files_merge_without_losing_precision(self):
        runs=[]
        for name in ['a.csv','b.csv']:
            runs.append(server.save_run('shm',[{'file_id':name,'prediction':0.12}],[],{'sha256':'one'},0,
                                        csv_content=f'file_id,prediction\n{name},0.12345678901234567\n'.encode()))
        merged=server.merge_runs([r['id'] for r in runs])
        self.assertEqual(len(merged['rows']),2)
        self.assertEqual(merged['combined_from'],[r['id'] for r in runs])
        self.assertIn(b'0.12345678901234567',merged['csv'])
        runs[0]['model']['sha256']='changed'
        with self.assertRaisesRegex(ValueError,'different models'):
            server.merge_runs([r['id'] for r in runs])

    def test_all_results_endpoint_uses_operator_headers(self):
        run=server.save_run('rail',[{'file_id':'a.csv','prediction':'Side I'}],[],{'sha256':'one'},0,
                            csv_content=b'file_id,prediction\na.csv,Side I\n')
        status, content=self.request('/api/all-results',json.dumps({'ids':[run['id']]}).encode())
        self.assertEqual(status,200)
        with zipfile.ZipFile(io.BytesIO(content)) as zipped:
            self.assertEqual(zipped.namelist(),['rail_check_results.csv'])
            self.assertIn(b'Rail corrugation result',zipped.read('rail_check_results.csv'))

    def test_preview_files_have_valid_schema(self):
        for key, config in server.SYSTEMS.items():
            with (server.ROOT / 'predictions' / config['output']).open() as file:
                rows = list(csv.DictReader(file))
            server.validate_rows(key, rows)
            self.assertEqual(server.csv_bytes(key, rows).decode().splitlines()[0], ','.join(config['columns']))


if __name__ == '__main__':
    unittest.main()
